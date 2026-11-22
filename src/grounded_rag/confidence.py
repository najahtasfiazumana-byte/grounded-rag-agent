"""Step 4: a confidence score that is explainable, not a model's self-rating.

    confidence = coverage_weight * question_coverage + (1 - coverage_weight) * grounded_share

question_coverage
    How much of the question is addressed by the best single cited source. Each key word of
    the question is weighted by how rare it is in the knowledge base (IDF), and a
    word that never appears in the knowledge base gets the highest weight. So for
    "latest version of Windows 11", a single match on "Windows" in a setup policy
    scores low: the important words ("latest", "version", "11") are missing.

grounded_share
    Share of the drafted sentences that passed verification.

If question_coverage is below min_coverage (50%) the score is capped at the
coverage itself, so it is always "low" and, when the evidence barely touches the
question, low enough to abstain. If the model said the sources don't contain
the answer, or nothing survived verification, confidence is 0.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

from .sources import Source
from .text import content_terms, expand_terms, is_number
from .verify import CheckedSentence


@dataclass
class Confidence:
    value: float
    level: str  # "high", "medium", "low"
    coverage: float
    grounded_share: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def level_for(value: float, high: float, medium: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


NUMBER_WEIGHT = 1.0  # numbers in a question are often the client's own figure ("a 12% discount")


def term_weight(term: str, idf: dict[str, float], unseen: float) -> float:
    return NUMBER_WEIGHT if is_number(term) else idf.get(term, unseen)


def weighted_coverage(question: str, evidence_text: str, idf: dict[str, float], n_docs: int,
                      glossary: Optional[dict] = None) -> float:
    q_terms = content_terms(question)
    if not q_terms:
        return 1.0
    unseen = math.log(1 + (n_docs + 0.5) / 0.5)  # weight for words the knowledge base never uses
    evidence = content_terms(evidence_text)
    matches = expand_terms(q_terms, glossary or {})
    total = sum(term_weight(t, idf, unseen) for t in q_terms)
    found = sum(term_weight(t, idf, unseen) for t in q_terms if matches[t] & evidence)
    return found / total if total else 0.0


def score(question: str, checked: list[CheckedSentence], sources: list[Source], *, answerable: bool,
          idf: dict[str, float], n_docs: int, coverage_weight: float = 0.6,
          high: float = 0.75, medium: float = 0.55, min_coverage: float = 0.5,
          glossary: Optional[dict] = None) -> Confidence:
    reasons: list[str] = []
    if not sources:
        return Confidence(0.0, "low", 0.0, 0.0, ["no relevant sources were found"])
    if not answerable or not checked:
        return Confidence(0.0, "low", 0.0, 0.0, ["the sources do not contain an answer"])

    kept = [c for c in checked if c.supported]
    grounded = len(kept) / len(checked)
    if not kept:
        return Confidence(0.0, "low", 0.0, 0.0, ["no sentence of the draft could be verified against its sources"])

    by_id = {s.id: s for s in sources}
    cited = {i for c in kept for i in c.sources if i in by_id}
    # Best SINGLE cited source, not the pooled text: two unrelated sources that each match half
    # the question ("reach ... support" in a hotline policy + "Windows ... 10" in a setup policy)
    # must not add up to a confident answer about Windows 10 end of support.
    coverage = max(weighted_coverage(question, f"{by_id[i].label} {by_id[i].text}", idf, n_docs, glossary)
                   for i in cited)
    value = round(coverage_weight * coverage + (1 - coverage_weight) * grounded, 3)

    if coverage < min_coverage:
        # Evidence that misses most of the question is never more than "low", however clean the
        # citations. Below 50% coverage the score is capped at the coverage itself, so evidence that
        # barely touches the question (say 25%) ends up in the "don't answer" zone.
        value = min(value, round(coverage, 3), round(medium - 0.01, 3))
        reasons.append(f"cited sources cover only {coverage:.0%} of what the question asks about")
    if grounded < 1:
        dropped = len(checked) - len(kept)
        reasons.append(f"{dropped} of {len(checked)} drafted sentence(s) failed the citation check and were removed")
    if not reasons:
        reasons.append("every sentence is backed by a cited source that covers the question")
    return Confidence(value, level_for(value, high, medium), round(coverage, 3), round(grounded, 3), reasons)
