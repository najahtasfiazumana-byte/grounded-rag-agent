"""A Source is anything an answer may cite: a knowledge-base chunk (S1, S2...) or a web result (W1, W2...)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from .index import Hit
from .websearch import WebResult


@dataclass
class Source:
    id: str  # "S1" for knowledge base, "W1" for web
    kind: str  # "kb" or "web"
    label: str  # human-readable: "Warranty and Returns Policy › Standard warranty" or page title
    ref: str  # chunk id or URL
    text: str
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def from_hits(hits: list[Hit]) -> list[Source]:
    return [Source(f"S{i}", "kb", h.chunk.label, h.chunk.id, h.chunk.text, round(h.score, 4))
            for i, h in enumerate(hits, start=1)]


def from_web(results: list[WebResult]) -> list[Source]:
    return [Source(f"W{i}", "web", r.title or r.url, r.url, r.snippet) for i, r in enumerate(results, start=1)]
