"""The agent: retrieve -> draft -> verify -> score -> (web fallback) -> answer, flag or abstain.

    question
       │
       ▼
    1. search the knowledge base (BM25, optionally + embeddings)
    2. draft an answer where every sentence cites sources       (LLM)
    3. verify each sentence against what it cites               (code)
    4. score confidence from coverage + verified share          (code)
       │
       ├── confidence >= fallback_below ──────────────► answer from the knowledge base
       │
       └── low ──► 5. live web search, then steps 2-4 again on the web results
                      │
                      ├── better result is >= fallback_below ─► answer (web or KB, whichever scored higher)
                      ├── between abstain_below and fallback ─► answer, flagged LOW CONFIDENCE
                      └── below abstain_below ────────────────► "I couldn't find a reliable answer"
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .answer import Draft, draft_answer
from .confidence import Confidence, score
from .config import Settings
from .index import Index
from .llm import LLM
from .sources import Source, from_hits, from_web
from .verify import CheckedSentence, verify_draft
from .text import load_glossary
from .websearch import NoSearch, SearchError, WebSearch

ABSTAIN_TEXT = (
    "I couldn't find a reliable answer to this in the knowledge base or on the web, "
    "so I won't guess. Please check with the relevant team."
)


@dataclass
class Attempt:
    route: str  # "kb" or "web"
    sources: list[Source]
    draft: Draft
    checked: list[CheckedSentence]
    confidence: Confidence


@dataclass
class AgentResult:
    question: str
    status: str  # "answered", "low_confidence", "abstained"
    route: str  # "kb", "web" or "none"
    confidence: Confidence
    sentences: list[CheckedSentence]  # verified sentences shown to the user
    removed: list[CheckedSentence]  # sentences dropped by verification
    sources: list[Source]  # sources available on the chosen route
    trace: list[dict] = field(default_factory=list)
    provider: str = ""
    web_provider: str = ""
    mode: str = "grounded"
    errors: list[str] = field(default_factory=list)  # model errors (rate limits, timeouts): not a real refusal
    asked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # -------------------------------------------------------------- presentation
    @property
    def cited_sources(self) -> list[Source]:
        """Sources in order of first citation; these become [1], [2], ... in the answer."""
        by_id = {s.id: s for s in self.sources}
        order: list[str] = []
        for sent in self.sentences:
            for sid in sent.sources:
                if sid in by_id and sid not in order:
                    order.append(sid)
        return [by_id[i] for i in order]

    def answer_text(self, with_citations: bool = True) -> str:
        if self.status == "abstained" or not self.sentences:
            return ABSTAIN_TEXT
        numbers = {s.id: n for n, s in enumerate(self.cited_sources, start=1)}
        parts = []
        for sent in self.sentences:
            marks = "".join(f"[{numbers[i]}]" for i in sent.sources if i in numbers)
            parts.append(f"{sent.text} {marks}".strip() if with_citations else sent.text)
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "asked_at": self.asked_at,
            "mode": self.mode,
            "status": self.status,
            "route": self.route,
            "answer": self.answer_text(),
            "confidence": self.confidence.to_dict(),
            "citations": [
                {"n": n, "id": s.id, "kind": s.kind, "label": s.label, "ref": s.ref}
                for n, s in enumerate(self.cited_sources, start=1)
            ],
            "sentences": [s.to_dict() for s in self.sentences],
            "removed": [s.to_dict() for s in self.removed],
            "sources": [s.to_dict() for s in self.sources],
            "trace": self.trace,
            "provider": self.provider,
            "web_provider": self.web_provider,
            "errors": self.errors,
        }

    def to_markdown(self) -> str:
        badge = {"answered": "✅ Answered", "low_confidence": "⚠️ LOW CONFIDENCE, verify before using",
                 "abstained": "🛑 Not answered"}[self.status]
        route = {"kb": "knowledge base", "web": "live web search", "none": "no source"}[self.route]
        lines = [f"**Q:** {self.question}", "",
                 f"{badge} · confidence {self.confidence.value:.2f} ({self.confidence.level}) · source: {route}", "",
                 self.answer_text(), ""]
        if self.cited_sources:
            lines.append("**Sources**")
            for n, s in enumerate(self.cited_sources, start=1):
                ref = f"<{s.ref}>" if s.kind == "web" else f"`{s.ref}`"
                lines.append(f"{n}. {s.label} ({ref})")
            lines.append("")
        if self.removed:
            lines.append("**Removed by the citation check**")
            for r in self.removed:
                lines.append(f"- ~~{r.text}~~ ({'; '.join(r.issues)})")
            lines.append("")
        lines.append("_Why this confidence: " + "; ".join(self.confidence.reasons) + "_")
        if self.errors:
            lines += ["", "**⚠️ Model error, so this is not a real refusal:** " + " | ".join(self.errors)]
        return "\n".join(lines)


class Agent:
    def __init__(self, index: Index, llm: LLM, search: Optional[WebSearch] = None,
                 settings: Optional[Settings] = None):
        self.index = index
        self.llm = llm
        self.settings = settings or Settings()
        self.search = search if (search is not None and self.settings.web_enabled) else NoSearch()
        self.glossary = load_glossary(self.settings.glossary_path)

    def _attempt(self, route: str, question: str, sources: list[Source], *, naive: bool) -> Attempt:
        s = self.settings
        idf = self.index.bm25.idf
        draft = draft_answer(question, sources, self.llm, naive=naive, glossary=self.glossary, weights=idf)
        checked = verify_draft(draft, sources, s.min_support)
        conf = score(question, checked, sources, answerable=draft.answerable, idf=idf,
                     n_docs=len(self.index.chunks), coverage_weight=s.coverage_weight,
                     high=s.high, medium=s.fallback_below, min_coverage=s.min_coverage,
                     glossary=self.glossary)
        if draft.error:
            conf.reasons.insert(0, f"model error: {draft.error}")
        return Attempt(route, sources, draft, checked, conf)

    def ask(self, question: str, *, mode: str = "grounded") -> AgentResult:
        """mode="grounded" is the real agent. mode="naive" is the baseline used by the eval:
        same retrieval and model, but no citation check, no confidence gate and no fallback."""
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")
        naive = mode == "naive"
        s, trace = self.settings, []

        t0 = time.perf_counter()
        hits = self.index.search(question, k=s.top_k, glossary=self.glossary)
        kb_sources = from_hits(hits)
        trace.append({"step": "retrieve", "ms": _ms(t0),
                      "detail": f"{len(hits)} knowledge-base section(s): " + ", ".join(h.chunk.id for h in hits)})

        t0 = time.perf_counter()
        kb = self._attempt("kb", question, kb_sources, naive=naive)
        trace.append({"step": "draft+verify (kb)", "ms": _ms(t0),
                      "detail": f"{len(kb.checked)} sentence(s), {sum(c.supported for c in kb.checked)} verified; "
                                f"confidence {kb.confidence.value:.2f} ({kb.confidence.level})"})

        errors = [kb.draft.error] if kb.draft.error else []
        if naive:
            # Baseline: show everything the model wrote, verified or not.
            status = "answered" if kb.draft.answerable and kb.checked else "abstained"
            return self._result(question, kb, status, trace, mode, keep_all=True, errors=errors)

        best = kb
        if kb.confidence.value < s.fallback_below and not isinstance(self.search, NoSearch):
            t0 = time.perf_counter()
            try:
                web_results = self.search.search(question, k=s.web_max_results)
                trace.append({"step": "web search", "ms": _ms(t0),
                              "detail": f"{self.search.name}: {len(web_results)} result(s)"})
            except SearchError as exc:
                web_results = []
                trace.append({"step": "web search", "ms": _ms(t0), "detail": f"failed: {exc}"})
            if web_results:
                t0 = time.perf_counter()
                web = self._attempt("web", question, from_web(web_results), naive=False)
                trace.append({"step": "draft+verify (web)", "ms": _ms(t0),
                              "detail": f"{len(web.checked)} sentence(s), "
                                        f"{sum(c.supported for c in web.checked)} verified; "
                                        f"confidence {web.confidence.value:.2f} ({web.confidence.level})"})
                if web.draft.error:
                    errors.append(web.draft.error)
                if web.confidence.value > kb.confidence.value:
                    best = web
        elif kb.confidence.value < s.fallback_below:
            trace.append({"step": "web search", "ms": 0, "detail": "skipped (web fallback is off)"})

        value = best.confidence.value
        if value < s.abstain_below or not any(c.supported for c in best.checked):
            status = "abstained"
        elif value < s.fallback_below:
            status = "low_confidence"
        else:
            status = "answered"
        trace.append({"step": "decide", "ms": 0, "detail": f"{status} from {best.route} (confidence {value:.2f})"})
        return self._result(question, best, status, trace, mode, errors=errors)

    def _result(self, question: str, att: Attempt, status: str, trace: list[dict], mode: str,
                keep_all: bool = False, errors: Optional[list[str]] = None) -> AgentResult:
        shown = att.checked if keep_all else [c for c in att.checked if c.supported]
        removed = [] if keep_all else [c for c in att.checked if not c.supported]
        route = att.route if status != "abstained" else "none"
        return AgentResult(question, status, route, att.confidence, shown, removed, att.sources, trace,
                           provider=self.llm.label, web_provider=self.search.name, mode=mode,
                           errors=list(errors or []))


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def write_audit(result: AgentResult, path: Path) -> None:
    """Append one JSON line per question so every answer can be traced later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "asked_at": result.asked_at,
        "question": result.question,
        "status": result.status,
        "route": result.route,
        "confidence": result.confidence.value,
        "answer": result.answer_text(),
        "citations": [s.ref for s in result.cited_sources],
        "removed_sentences": len(result.removed),
        "provider": result.provider,
        "web_provider": result.web_provider,
        "errors": result.errors,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
