"""Load a folder of documents and split them into citable chunks.

Supported: Markdown (.md), plain text (.txt) and PDF (.pdf, needs `pypdf`).
Markdown is split on headings, so every chunk maps to one section a person
can open and check. Long sections are split again by paragraph with a small
overlap. Every chunk gets a stable, human-readable id such as
`warranty_and_returns#returns-and-restocking-fee`, which is what the answer cites.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .text import slugify

MAX_WORDS = 180  # sections longer than this are split by paragraph


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    source_path: str

    @property
    def label(self) -> str:
        return f"{self.doc_title} › {self.section}" if self.section else self.doc_title

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


def _clean(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):  # blockquote disclaimers are not facts
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _split_long(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if sum(len(p.split()) for p in paragraphs) <= MAX_WORDS or len(paragraphs) == 1:
        return [text]
    parts, current = [], []
    for para in paragraphs:
        if current and sum(len(p.split()) for p in current) + len(para.split()) > MAX_WORDS:
            parts.append("\n\n".join(current))
            current = [current[-1]]  # one paragraph of overlap keeps context across the cut
        current.append(para)
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_markdown(text: str, doc_id: str, source_path: str) -> list[Chunk]:
    title = doc_id.replace("_", " ").title()
    sections: list[tuple[str, list[str]]] = []
    heading, body = "", []
    for line in text.splitlines():
        if line.startswith("# ") and not sections and not body:
            title = line[2:].strip()
            continue
        match = re.match(r"^#{2,4}\s+(.*)", line)
        if match:
            if "".join(body).strip():
                sections.append((heading, body))
            heading, body = match.group(1).strip(), []
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((heading, body))

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for heading, lines in sections:
        cleaned = _clean("\n".join(lines))
        if not cleaned:
            continue
        for n, part in enumerate(_split_long(cleaned), start=1):
            base = f"{doc_id}#{slugify(heading) if heading else 'intro'}"
            cid = base if n == 1 else f"{base}-{n}"
            while cid in seen:
                cid += "x"
            seen.add(cid)
            chunks.append(Chunk(cid, doc_id, title, heading, part, source_path))
    return chunks


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Reading PDFs needs pypdf: pip install pypdf") from exc
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(f"## Page {i}\n\n{page.extract_text() or ''}")
    return "\n\n".join(pages)


def load_folder(folder: Path) -> list[Chunk]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {folder}")
    chunks: list[Chunk] = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"} or not path.is_file():
            continue
        doc_id = slugify(path.stem).replace("-", "_")
        text = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
        rel = path.relative_to(folder.parent) if folder.parent in path.parents else path
        chunks.extend(chunk_markdown(text, doc_id, rel.as_posix()))
    if not chunks:
        raise ValueError(f"No .md, .txt or .pdf documents found in {folder}")
    return chunks
