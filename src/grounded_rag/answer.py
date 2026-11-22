"""Step 2: write a draft answer where every sentence names the sources it came from.

With a real model (Gemini, Claude, Ollama) the model writes the sentences and
picks citations. With the offline "mock" provider an extractive answerer copies
the source sentences that best match the question, so the whole pipeline runs
free with no key.

The draft is NOT trusted. verify.py checks every sentence afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .llm import LLM, LLMError, QuotaExhausted
from .sources import Source
from .text import content_terms, expand_terms, is_number, split_sentences

MAX_SENTENCES = 4

GROUNDED_SYSTEM = """You answer questions for a B2B IT sales team using ONLY the numbered sources provided.

Rules:
1. Every sentence must list the id(s) of the source(s) that support it, e.g. ["S2"].
2. Use only facts stated in the sources. Copy numbers, prices, percentages and time periods exactly. Keep wording close to the source.
3. Do not use outside knowledge, and do not guess. If the sources do not contain the answer, set "answerable" to false and return no sentences.
4. Only cite a source if it directly supports that sentence.
5. Answer in at most 4 short sentences, most important fact first.

Return a JSON object only:
{"answerable": true, "sentences": [{"text": "...", "sources": ["S1"]}]}"""

NAIVE_SYSTEM = """You are a helpful assistant for a B2B IT sales team. Answer the question.
Some context is provided and may help. Answer in at most 4 short sentences.

Return a JSON object only:
{"answerable": true, "sentences": [{"text": "...", "sources": []}]}"""


@dataclass
class Draft:
    answerable: bool
    sentences: list[dict] = field(default_factory=list)  # [{"text": str, "sources": [ids]}]
    method: str = "llm"  # "llm" or "extractive"
    error: Optional[str] = None


def build_user_prompt(question: str, sources: list[Source]) -> str:
    blocks = [f"[{s.id}] ({s.label})\n{s.text}" for s in sources]
    return f"Question: {question}\n\nSources:\n\n" + "\n\n".join(blocks)


def _clean_sentences(raw: list, valid_ids: set[str]) -> list[dict]:
    cleaned = []
    for item in raw or []:
        if isinstance(item, str):
            item = {"text": item, "sources": []}
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ids = item.get("sources") or item.get("citations") or []
        if isinstance(ids, str):
            ids = [ids]
        ids = [str(i).strip().strip("[]").upper() for i in ids]
        cleaned.append({"text": text, "sources": [i for i in dict.fromkeys(ids) if i]})
    return cleaned[: MAX_SENTENCES + 2]


def extractive_answer(question: str, sources: list[Source], glossary: Optional[dict] = None,
                      weights: Optional[dict[str, float]] = None) -> Draft:
    """Offline answerer: copy the best-matching source sentences, citing where each came from.

    Each question word counts by its weight (rarer words count more). The section heading
    counts at half weight, so "when must we send the renewal quote" prefers sentences
    in the "Renewal timeline" section.
    """
    q_terms = content_terms(question)
    if not q_terms:
        return Draft(False, method="extractive")
    matches = expand_terms(q_terms, glossary or {})
    raw_weights = weights or {}
    # Numbers in a question are usually the client's own figure ("a 12% discount"), so they count little.
    weights = {t: (0.5 if is_number(t) else raw_weights.get(t, 1.0)) for t in q_terms}
    total = sum(weights.values())
    candidates = []
    for rank, src in enumerate(sources):
        heading = content_terms(src.label)
        for pos, sentence in enumerate(split_sentences(src.text)):
            terms = content_terms(sentence)
            score = sum(weights.get(t, 1.0) for t in q_terms if matches[t] & terms)
            score += 0.5 * sum(weights.get(t, 1.0) for t in q_terms if matches[t] & heading and not matches[t] & terms)
            if score:
                # Prefer higher score, then higher-ranked sources, then earlier sentences.
                candidates.append((round(score / total, 6), -rank, -pos, sentence, src.id))
    if not candidates:
        return Draft(False, method="extractive")
    candidates.sort(reverse=True)
    best = candidates[0][0]
    chosen = [c for c in candidates if c[0] >= best * 0.6][:2]
    return Draft(True, [{"text": c[3], "sources": [c[4]]} for c in chosen], method="extractive")


def draft_answer(question: str, sources: list[Source], llm: LLM, *, naive: bool = False,
                 glossary: Optional[dict] = None, weights: Optional[dict[str, float]] = None) -> Draft:
    if not sources:
        return Draft(False, method="none")
    if llm.name == "mock":
        return extractive_answer(question, sources, glossary, weights)
    system = NAIVE_SYSTEM if naive else GROUNDED_SYSTEM
    try:
        data = llm.complete_json(system, build_user_prompt(question, sources))
    except QuotaExhausted:
        raise  # no point carrying on: every following call would fail too
    except LLMError as exc:
        return Draft(False, error=str(exc)[:300])
    sentences = _clean_sentences(data.get("sentences", []), {s.id for s in sources})
    answerable = bool(data.get("answerable", bool(sentences))) and bool(sentences)
    return Draft(answerable, sentences if answerable else [])
