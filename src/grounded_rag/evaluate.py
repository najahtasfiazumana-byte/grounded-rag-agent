"""Measure what the guardrails actually buy: naive RAG vs the grounded agent, same questions, same model.

naive     retrieve top-k -> model answers -> everything is shown. No citation check,
          no confidence gate, no web fallback. This is a typical "first RAG demo".
grounded  the full agent: citation check, confidence score, web fallback, abstain.

Question types (data/eval/questions.jsonl):
  kb            answer is in the knowledge base; `expect_all` strings must appear in the answer
  web           answer is NOT in the knowledge base but is on the public web
  unanswerable  nobody should answer (private or non-existent data); the right move is to abstain

A shown answer counts as HALLUCINATED when it is shown without a low-confidence flag and
  - the question is unanswerable, or
  - it is missing the expected fact (kb/web), or
  - any shown sentence fails the citation check (no source, wrong number, unsupported wording).

A question where EITHER pipeline hit a model error (rate limit, timeout) is marked "error" for both
and left out of every rate, so a quota problem can never look like a safe refusal. Both pipelines
run back to back on each question, so any problem hits them equally.

Note: with the offline "mock" provider the numbers show that the pipeline works, not how a real
model behaves. Run `rag-agent eval --provider gemini` to measure with a real model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from typing import Callable, Optional

from .agent import Agent, AgentResult
from .llm import QuotaExhausted
from .text import numbers_in


@dataclass
class EvalCase:
    id: str
    type: str
    question: str
    expect_all: list[str]
    expect_docs: list[str]


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cases.append(EvalCase(d["id"], d["type"], d["question"], d.get("expect_all", []), d.get("expect_docs", [])))
    return cases


def _contains(answer: str, expected: str) -> bool:
    a, e = answer.lower(), expected.lower()
    if e in a:
        return True
    # "MYR 3000" vs "3,000": compare numbers without separators.
    nums = numbers_in(expected)
    return bool(nums) and nums <= numbers_in(answer) and e.replace(",", "") in a.replace(",", "")


def judge(case: EvalCase, r: AgentResult) -> dict:
    shown = r.status != "abstained"
    flagged = r.status == "low_confidence"
    unsupported = sum(1 for s in r.sentences if not s.supported)
    answer = r.answer_text(with_citations=False) if shown else ""
    has_fact = all(_contains(answer, e) for e in case.expect_all) if case.expect_all else None
    cited_docs = {s.ref.split("#")[0] for s in r.cited_sources if s.kind == "kb"}
    right_doc = bool(cited_docs & set(case.expect_docs)) if case.expect_docs else None

    if r.errors:
        outcome = "error"
    elif not shown:
        outcome = "abstained"
    elif case.type == "unanswerable":
        outcome = "flagged" if flagged else "hallucinated"
    elif unsupported or not has_fact:
        outcome = "flagged" if flagged else "hallucinated"
    else:
        outcome = "correct_flagged" if flagged else "correct"

    return {
        "id": case.id, "type": case.type, "question": case.question, "status": r.status, "route": r.route,
        "confidence": r.confidence.value, "outcome": outcome, "has_expected_fact": has_fact,
        "cited_expected_doc": right_doc, "sentences_shown": len(r.sentences) if shown else 0,
        "unsupported_shown": unsupported if shown else 0, "answer": r.answer_text() if shown else "",
        "errors": r.errors,
    }


def summarise(all_rows: list[dict]) -> dict:
    rows = [r for r in all_rows if r["outcome"] != "error"]
    n = len(rows)
    by_type = {t: [r for r in rows if r["type"] == t] for t in ("kb", "web", "unanswerable")}
    shown = sum(r["sentences_shown"] for r in rows)

    def rate(items, pred):
        return round(sum(1 for r in items if pred(r)) / len(items), 3) if items else None

    return {
        "questions": n,
        "errors": len(all_rows) - n,
        "hallucinated_answer_rate": rate(rows, lambda r: r["outcome"] == "hallucinated"),
        "kb_correct_rate": rate(by_type["kb"], lambda r: r["outcome"] in ("correct", "correct_flagged")),
        "kb_cited_right_doc_rate": rate([r for r in by_type["kb"] if r["status"] != "abstained"],
                                        lambda r: r["cited_expected_doc"]),
        "web_correct_rate": rate(by_type["web"], lambda r: r["outcome"] in ("correct", "correct_flagged")),
        "unanswerable_abstain_or_flag_rate": rate(by_type["unanswerable"],
                                                  lambda r: r["outcome"] in ("abstained", "flagged")),
        "unsupported_sentence_rate": round(sum(r["unsupported_shown"] for r in rows) / shown, 3) if shown else 0.0,
        "flagged_rate": rate(rows, lambda r: r["status"] == "low_confidence"),
        "abstain_rate": rate(rows, lambda r: r["status"] == "abstained"),
    }


def run_eval(agent: Agent, cases: list[EvalCase], out_dir: Path, *, delay: float = 0.0,
             progress: Optional[Callable[[str], None]] = None) -> dict:
    report = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "provider": agent.llm.label,
              "web_provider": agent.search.name, "modes": {}, "stopped_early": None}
    rows: dict[str, list[dict]] = {"naive": [], "grounded": []}
    for i, case in enumerate(cases, start=1):
        try:
            pair = {}
            for mode in ("naive", "grounded"):
                pair[mode] = judge(case, agent.ask(case.question, mode=mode))
                if delay:
                    time.sleep(delay)
        except QuotaExhausted as exc:
            report["stopped_early"] = f"after {i - 1} of {len(cases)} questions: {exc}"
            if progress:
                progress(f"STOPPED: {exc}")
            break
        if pair["naive"]["outcome"] == "error" or pair["grounded"]["outcome"] == "error":
            for row in pair.values():  # paired: exclude the question from both pipelines
                row["outcome"] = "error"
        for mode, row in pair.items():
            rows[mode].append(row)
        if progress:
            progress(f"[{i:>2}/{len(cases)}] {case.id:<7} naive: {pair['naive']['outcome']:<15} "
                     f"grounded: {pair['grounded']['outcome']}")

    for mode in ("naive", "grounded"):
        report["modes"][mode] = {"summary": summarise(rows[mode]), "rows": rows[mode]}
    n, g = report["modes"]["naive"]["summary"], report["modes"]["grounded"]["summary"]
    before, after = n["hallucinated_answer_rate"], g["hallucinated_answer_rate"]
    report["hallucination_reduction"] = round((before - after) / before, 3) if before else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "eval_report.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def is_trustworthy(report: dict) -> bool:
    """True when every question completed in both pipelines."""
    n = report["modes"]["naive"]["summary"]
    return not report.get("stopped_early") and not n.get("errors") and n["questions"] > 0


METRICS = [
    ("hallucinated_answer_rate", "Hallucinated answers shown (lower is better)"),
    ("unsupported_sentence_rate", "Shown sentences with no valid evidence (lower is better)"),
    ("kb_correct_rate", "Knowledge-base questions answered correctly"),
    ("kb_cited_right_doc_rate", "...and citing the right document"),
    ("web_correct_rate", "Web-only questions answered correctly (via fallback)"),
    ("unanswerable_abstain_or_flag_rate", "Unanswerable questions refused or flagged"),
    ("flagged_rate", "Answers flagged low confidence"),
    ("abstain_rate", "Questions refused (\"I won't guess\")"),
]


def to_markdown(report: dict) -> str:
    n, g = report["modes"]["naive"]["summary"], report["modes"]["grounded"]["summary"]
    lines = [
        "# Evaluation: naive RAG vs grounded agent", "",
        f"- Run: {report['run_at']}",
        f"- Model: `{report['provider']}` · web search: `{report['web_provider']}` · questions: {n['questions']}",
    ]
    if report["provider"].startswith("mock"):
        lines.append("- ⚠️ Offline mock model: these numbers show the pipeline works. "
                     "Run with `--provider gemini` for real-model numbers.")
    if n.get("errors"):
        lines.append(f"- ⚠️ **{n['errors']} question(s) hit model errors** (rate limit or timeout) and are left out "
                     "of every rate. Re-run with `--delay 6` for complete numbers.")
    if report.get("stopped_early"):
        lines.append(f"- 🛑 **Stopped early** {report['stopped_early']}")
    if n["questions"] == 0:
        lines.append("- No question completed, so there are no numbers to report.")
    red = report.get("hallucination_reduction")
    if red is not None:
        lines.append(f"- **Hallucinated answers reduced by {red:.0%}** "
                     f"({_pct(n['hallucinated_answer_rate'])} → {_pct(g['hallucinated_answer_rate'])})")
    lines += ["", "| Metric | Naive RAG | Grounded agent |", "|---|---:|---:|"]
    for key, label in METRICS:
        lines.append(f"| {label} | {_pct(n[key])} | {_pct(g[key])} |")
    lines += ["", "## Per question (grounded agent)", "", "| id | type | status | route | conf. | outcome |",
              "|---|---|---|---|---:|---|"]
    for r in report["modes"]["grounded"]["rows"]:
        lines.append(f"| {r['id']} | {r['type']} | {r['status']} | {r['route']} | {r['confidence']:.2f} | {r['outcome']} |")
    lines += ["", "Definitions are in `src/grounded_rag/evaluate.py`.", ""]
    return "\n".join(lines)
