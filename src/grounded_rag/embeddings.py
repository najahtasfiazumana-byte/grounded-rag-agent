"""Optional dense (embedding) retrieval with Gemini's free embedding model.

Keyword search (BM25) is the default because it needs no key and is exact.
Turning embeddings on adds meaning-based matching ("laptop for a designer"
finds the section about "designers and engineers"). The two ranked lists are
merged with Reciprocal Rank Fusion, a simple and robust way to combine them.
"""

from __future__ import annotations

import math
import os
from typing import Optional


from .llm import LLMError, _post_with_retry

DEFAULT_EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
_BATCH = 50


class GeminiEmbedder:
    name = "gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.getenv("GEMINI_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set, so Gemini embeddings cannot be used")

    def _headers(self) -> dict:
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            batch = texts[start : start + _BATCH]
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": "RETRIEVAL_DOCUMENT",
                        "outputDimensionality": EMBED_DIM,
                    }
                    for t in batch
                ]
            }
            data = _post_with_retry(url, headers=self._headers(), payload=payload)
            vectors.extend(_normalise(e["values"]) for e in data["embeddings"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": EMBED_DIM,
        }
        data = _post_with_retry(url, headers=self._headers(), payload=payload)
        return _normalise(data["embedding"]["values"])


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Vectors are stored normalised, so cosine similarity is a dot product."""
    return sum(x * y for x, y in zip(a, b))


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def get_embedder(name: str):
    if name in ("", "none", None):
        return None
    if name == "gemini":
        return GeminiEmbedder()
    raise ValueError(f"Unknown embeddings provider {name!r}. Use 'none' or 'gemini'.")


