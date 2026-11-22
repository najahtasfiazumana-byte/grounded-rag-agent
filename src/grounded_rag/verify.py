"""Step 3: check every sentence against the sources it cites. Code decides, not the model.

A sentence is kept only if:
  1. it cites at least one source, and every cited id is one we actually gave the model;
  2. every number in it (prices, %, days, counts) appears in a cited source,
     with "15%" and "15" treated as different facts; and
  3. enough of its key words appear in the cited source(s) (min_support, default 50%).

Sentences that fail are removed from the answer and shown separately with the reason,
so a reviewer can see exactly what the model tried to say without evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .answer import Draft
from .sources import Source
from .text import content_terms, numbers_in


@dataclass
class CheckedSentence:
    text: str
    sources: list[str]
    supported: bool
    support: float  # share of the sentence's key words found in its cited sources (0-1)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def check_sentence(text: str, cited: list[str], sources: dict[str, Source], min_support: float) -> CheckedSentence:
    issues: list[str] = []
    known = [c for c in cited if c in sources]
    unknown = [c for c in cited if c not in sources]
    if unknown:
        issues.append(f"cites source(s) that were not provided: {', '.join(unknown)}")
    if not known:
        if not cited:
            issues.append("no citation")
        return CheckedSentence(text, cited, False, 0.0, issues)

    evidence = " ".join(sources[c].text for c in known)
    evidence_terms = content_terms(evidence)
    evidence_numbers = numbers_in(evidence)

    missing_numbers = sorted(numbers_in(text) - evidence_numbers)
    if missing_numbers:
        issues.append(f"number(s) not in the cited source: {', '.join(missing_numbers)}")

    terms = content_terms(text)
    support = 1.0 if not terms else len(terms & evidence_terms) / len(terms)
    if support < min_support:
        issues.append(f"only {support:.0%} of its key words appear in the cited source")

    supported = not issues
    return CheckedSentence(text, cited, supported, round(support, 3), issues)


def verify_draft(draft: Draft, sources: list[Source], min_support: float = 0.5) -> list[CheckedSentence]:
    by_id = {s.id: s for s in sources}
    return [check_sentence(s["text"], s.get("sources", []), by_id, min_support) for s in draft.sentences]
