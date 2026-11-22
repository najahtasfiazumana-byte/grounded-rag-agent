# Evaluation: naive RAG vs grounded agent

- Run: 2026-11-22T01:26:37+00:00
- Model: `mock` · web search: `mock` · questions: 25
- ⚠️ Offline mock model: these numbers show the pipeline works. Run with `--provider gemini` for real-model numbers.
- **Hallucinated answers reduced by 90%** (40% → 4%)

| Metric | Naive RAG | Grounded agent |
|---|---:|---:|
| Hallucinated answers shown (lower is better) | 40% | 4% |
| Shown sentences with no valid evidence (lower is better) | 0% | 0% |
| Knowledge-base questions answered correctly | 94% | 94% |
| ...and citing the right document | 100% | 100% |
| Web-only questions answered correctly (via fallback) | 0% | 67% |
| Unanswerable questions refused or flagged | 0% | 100% |
| Answers flagged low confidence | 0% | 8% |
| Questions refused ("I won't guess") | 0% | 20% |

## Per question (grounded agent)

| id | type | status | route | conf. | outcome |
|---|---|---|---|---:|---|
| kb01 | kb | answered | kb | 1.00 | correct |
| kb02 | kb | answered | kb | 1.00 | correct |
| kb03 | kb | answered | kb | 1.00 | correct |
| kb04 | kb | answered | kb | 1.00 | correct |
| kb05 | kb | answered | kb | 1.00 | correct |
| kb06 | kb | answered | kb | 0.92 | correct |
| kb07 | kb | answered | kb | 1.00 | correct |
| kb08 | kb | answered | kb | 1.00 | correct |
| kb09 | kb | answered | kb | 1.00 | correct |
| kb10 | kb | low_confidence | kb | 0.50 | flagged |
| kb11 | kb | answered | kb | 1.00 | correct |
| kb12 | kb | answered | kb | 1.00 | correct |
| kb13 | kb | answered | kb | 1.00 | correct |
| kb14 | kb | answered | kb | 1.00 | correct |
| kb15 | kb | answered | kb | 1.00 | correct |
| kb16 | kb | answered | kb | 0.84 | correct |
| web01 | web | answered | web | 1.00 | correct |
| web02 | web | answered | web | 0.85 | correct |
| web03 | web | answered | kb | 1.00 | hallucinated |
| none01 | unanswerable | abstained | none | 0.08 | abstained |
| none02 | unanswerable | abstained | none | 0.14 | abstained |
| none03 | unanswerable | abstained | none | 0.30 | abstained |
| none04 | unanswerable | abstained | none | 0.44 | abstained |
| none05 | unanswerable | low_confidence | kb | 0.45 | flagged |
| none06 | unanswerable | abstained | none | 0.36 | abstained |

Definitions are in `src/grounded_rag/evaluate.py`.
