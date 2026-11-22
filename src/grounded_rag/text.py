"""Small, dependency-free text helpers shared by retrieval and verification.

The same tokenizer is used to index documents, to read the question and to
check whether an answer sentence is supported by its cited source. Keeping
one tokenizer everywhere means "supported" has one consistent meaning.
"""

from __future__ import annotations

import re

# Ordinary English stopwords plus words that appear in questions but carry no
# meaning for retrieval ("how long is the warranty" -> "warranty").
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its itself just me more most my no nor
    not of off on once only or other our ours out over own same she should so some such than that the
    their theirs them then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours

    tell please need know get got give want wants much many long anything something someone our us
    kiara tech company
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")
_CITATION_RE = re.compile(r"\[(?:S|W)\d+\]", re.I)
# Same number pattern as the quote and renewal projects: "15%" and "15" are different facts.
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s?%?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


# Irregular forms that a suffix stemmer can't catch.
IRREGULAR = {"sent": "send", "paid": "pay", "bought": "buy", "sold": "sell", "kept": "keep",
             "won": "win", "began": "begin", "left": "leave", "made": "make"}


def stem(word: str) -> str:
    """A deliberately tiny stemmer: enough to match 'returns'/'return', 'days'/'day',
    'approves'/'approval'/'approved' and 'sent'/'send'."""
    if word.endswith("'s"):
        word = word[:-2]
    if word in IRREGULAR:
        return IRREGULAR[word]
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if len(word) > 5 and word.endswith("al"):
        word = word[:-2]
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase, split into words/numbers, drop stopwords, stem."""
    text = _CITATION_RE.sub(" ", text.lower()).replace("’", "'")
    tokens = []
    for raw in _TOKEN_RE.findall(text):
        raw = raw.rstrip(".,")
        if not raw:
            continue
        if not keep_stopwords and raw in STOPWORDS:
            continue
        tokens.append(raw.replace(",", "") if raw[0].isdigit() else stem(raw))
    return tokens


def content_terms(text: str) -> set[str]:
    return set(tokenize(text))


def is_number(term: str) -> bool:
    return term[:1].isdigit()


def expand_terms(terms: set[str], glossary: dict[str, list[str]]) -> dict[str, set[str]]:
    """Map each question term to the set of words that count as a match for it.

    With the glossary entry `m365: [microsoft 365]`, the term "m365" is also matched by
    "microsoft" + "365", so "M365" in a question finds "Microsoft 365" in a document.
    """
    out: dict[str, set[str]] = {}
    for t in terms:
        alts = {t}
        for phrase in glossary.get(t, []):
            alts |= set(tokenize(phrase))
        out[t] = alts
    return out


def load_glossary(path) -> dict[str, list[str]]:
    """Glossary file: YAML mapping of abbreviation/jargon -> list of plain phrases."""
    from pathlib import Path

    import yaml

    path = Path(path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    glossary: dict[str, list[str]] = {}
    for key, phrases in raw.items():
        phrases = [phrases] if isinstance(phrases, str) else list(phrases or [])
        for tok in tokenize(str(key)):
            glossary.setdefault(tok, []).extend(str(p) for p in phrases)
    return glossary


def normalise_number(raw: str) -> str:
    """'MYR 3,000' and '3000' are the same number; '15%' and '15 %' are the same percentage."""
    return raw.replace(",", "").replace(" ", "")


def numbers_in(text: str) -> set[str]:
    return {normalise_number(m.group(0)) for m in NUMBER_RE.finditer(_CITATION_RE.sub(" ", text))}


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
