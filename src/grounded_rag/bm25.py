"""BM25 keyword search in plain Python (no numpy, no vector database).

BM25 is the ranking function behind most search engines. For a few hundred
or a few thousand policy chunks it is fast, free and very good at matching
exact terms such as product names, "SST" or "Net 30", which pure embedding
search often misses.
"""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = [Counter(tokens) for tokens in docs_tokens]
        self.doc_lens = [len(tokens) for tokens in docs_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        n = len(docs_tokens)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def score(self, query_tokens: list[str], index: int) -> float:
        freqs, dl = self.doc_freqs[index], self.doc_lens[index]
        total = 0.0
        for term in set(query_tokens):
            tf = freqs.get(term)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
        return total

    def search(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        scored = [(i, self.score(query_tokens, i)) for i in range(len(self.doc_freqs))]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: (-s[1], s[0]))
        return scored[:k]
