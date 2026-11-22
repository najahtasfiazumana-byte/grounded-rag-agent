"""The searchable index: chunks + BM25 (+ optional embeddings), saved as one JSON file."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .bm25 import BM25
from .documents import Chunk, load_folder
from .embeddings import cosine, get_embedder, reciprocal_rank_fusion
from .text import expand_terms, tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float  # final score used for ranking (BM25, or fused RRF score in hybrid mode)
    bm25: float
    dense: Optional[float] = None


def _index_tokens(chunk: Chunk) -> list[str]:
    # The section heading and document title are repeated so they weigh more than body text.
    return tokenize(f"{chunk.doc_title} {chunk.section} {chunk.section} {chunk.text}")


def fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]


class Index:
    def __init__(self, chunks: list[Chunk], vectors: Optional[list[list[float]]] = None,
                 embedder_name: str = "none", built_at: Optional[str] = None, kb_path: str = ""):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder_name = embedder_name if vectors else "none"
        self.built_at = built_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.kb_path = kb_path
        self.bm25 = BM25([_index_tokens(c) for c in chunks])
        self._embedder = None
        self.by_id = {c.id: c for c in chunks}

    # ------------------------------------------------------------------ build / persist
    @classmethod
    def build(cls, kb_folder: Path, embeddings: str = "none") -> "Index":
        chunks = load_folder(Path(kb_folder))
        vectors = None
        embedder = get_embedder(embeddings)
        if embedder is not None:
            vectors = embedder.embed_documents([f"{c.label}\n{c.text}" for c in chunks])
        return cls(chunks, vectors, embeddings, kb_path=Path(kb_folder).as_posix())

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "built_at": self.built_at,
            "kb_path": self.kb_path,
            "fingerprint": fingerprint(self.chunks),
            "embedder": self.embedder_name,
            "chunks": [c.to_dict() for c in self.chunks],
            "vectors": self.vectors,
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        return cls(chunks, data.get("vectors"), data.get("embedder", "none"), data.get("built_at"),
                   data.get("kb_path", ""))

    @classmethod
    def load_or_build(cls, index_path: Path, kb_folder: Path, embeddings: str = "none") -> "Index":
        """Reuse the saved index unless the documents changed since it was built."""
        index_path, kb_folder = Path(index_path), Path(kb_folder)
        if index_path.exists():
            idx = cls.load(index_path)
            try:
                current = fingerprint(load_folder(kb_folder))
            except (FileNotFoundError, ValueError):
                return idx
            if current == fingerprint(idx.chunks) and (embeddings == "none" or idx.embedder_name == embeddings):
                return idx
        idx = cls.build(kb_folder, embeddings)
        idx.save(index_path)
        return idx

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 5, glossary: Optional[dict] = None) -> list[Hit]:
        q_tokens = tokenize(query)
        if glossary:
            for alts in expand_terms(set(q_tokens), glossary).values():
                q_tokens.extend(sorted(alts - set(q_tokens)))
        bm25_hits = self.bm25.search(q_tokens, k=max(k * 3, 10))
        bm25_scores = dict(bm25_hits)

        if not self.vectors:
            return [Hit(self.chunks[i], s, s) for i, s in bm25_hits[:k]]

        if self._embedder is None:
            self._embedder = get_embedder(self.embedder_name)
        q_vec = self._embedder.embed_query(query)
        dense = sorted(((i, cosine(q_vec, v)) for i, v in enumerate(self.vectors)), key=lambda x: -x[1])
        dense_scores = dict(dense)
        fused = reciprocal_rank_fusion([[i for i, _ in bm25_hits], [i for i, _ in dense[: max(k * 3, 10)]]])
        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [Hit(self.chunks[i], s, bm25_scores.get(i, 0.0), dense_scores.get(i)) for i, s in ranked]
