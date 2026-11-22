# Architecture and design decisions

## Pipeline

```
question
  │
  ├─ 1. retrieve      index.py      BM25 over heading-sized chunks (+ optional embeddings, fused with RRF)
  │                                 query expanded with data/glossary.yaml (M365 -> Microsoft 365)
  │
  ├─ 2. draft         answer.py     LLM returns {"answerable": bool, "sentences": [{"text", "sources": ["S1"]}]}
  │                                 offline: extractive answerer copies best-matching source sentences
  │
  ├─ 3. verify        verify.py     per sentence: valid citation? numbers in source? >= 50% key words in source?
  │
  ├─ 4. score         confidence.py 0.6 x question coverage (best single cited source, IDF-weighted)
  │                                 + 0.4 x share of sentences that passed verification
  │                                 coverage < 50% -> score capped at coverage (always "low")
  │
  ├─ 5. fallback      websearch.py  if score < 0.55: search the web, repeat 2-4 on W1..Wn, keep the better attempt
  │
  └─ 6. decide        agent.py      >= 0.55 answer · 0.45-0.55 answer + LOW CONFIDENCE flag · < 0.45 refuse
                                    every question appended to outputs/audit.jsonl
```

## Decisions

### The model writes; code decides
The LLM is good at reading sources and writing a clear sentence. It is not reliable at judging its own
accuracy: self-reported confidence is poorly calibrated. So the model's only jobs are to draft and to
say which source each sentence came from. Whether a sentence is shown, and how confident the answer is,
are decided by deterministic code that can be unit-tested. This is the same pattern as
[quote-agent](https://github.com/najahtasfiazumana-byte/quote-agent), where the LLM extracts and
`Decimal` does the maths.

### Citations per sentence, not per answer
A single "Sources: [1][2][3]" list at the bottom can't be checked: you don't know which claim came from
where. Per-sentence citations make each claim independently verifiable, by code and by the rep.

### Number check is strict and percent-aware
Numbers are where a wrong answer costs money (a fee, a discount, a deadline). Any number in a sentence
must appear in its cited source. `15%` and `15` are different facts, so "15% off" can't be justified by
"MYR 15". Thousands separators are ignored, so `MYR 3,000` matches `3000`. The same regex is used in the
quote and renewal projects.

### Coverage from the best single source
An early version pooled all cited sources. For "When did Windows 10 reach end of support?" it cited a
hotline policy ("clients *reach* the *support* hotline") and a setup policy ("*Windows* ... *10*
devices"). Together they "covered" the question, so the agent answered confidently and wrongly. Scoring
coverage on the best single cited source fixed it: the agent now falls back to the web. A test
(`test_off_topic_evidence_is_capped_low`) keeps it fixed.

### IDF weighting
Not all question words matter equally. "What is the latest version of Windows 11?" shares the word
"Windows" with a setup policy, but the important words ("latest", "version", "11") appear nowhere in the
playbook. Words are weighted by how rare they are in the knowledge base, and words the knowledge base
never uses get the highest weight. Numbers in the question get a low fixed weight, because they are
usually the client's own figure ("a 12% discount") rather than something the policy states.

### BM25 first, embeddings optional
Sales questions are full of exact terms: product names, "Net 30", "SST", "DOA". BM25 matches them
exactly, costs nothing and needs no key, so CI runs fully offline. Embeddings help with paraphrases
("laptop for a designer"), so they're available as an option and fused with BM25 using Reciprocal Rank
Fusion. RRF combines ranks rather than raw scores, so no score calibration is needed.

### Heading-sized chunks
Markdown is split on `##` headings. A citation then points to a section a person can open and read in
10 seconds, such as `warranty_and_returns#returns-and-restocking-fee`. Sections over 180 words are split
by paragraph with one paragraph of overlap.

### Web fallback goes through the same checks
Web results become sources `W1..Wn` and pass through exactly the same draft, verify and score steps. A
web answer is only preferred when it scores higher than the knowledge-base attempt. Search failures
(rate limits, no internet) are recorded in the trace and the agent degrades to flag or refuse. It never
crashes.

### Refusing is a feature
For a sales team, "I don't know, check with finance" is cheap, and a confident wrong answer is
expensive. The eval counts a refused or flagged answer as safe and an unflagged wrong answer as a
hallucination.

## Evaluation definitions (`evaluate.py`)

| Outcome | Meaning |
|---|---|
| `correct` | Shown, unflagged, contains the expected fact, all sentences verified |
| `correct_flagged` | Same, but flagged low confidence |
| `flagged` | Wrong, unsupported or unanswerable, but shown with the low-confidence flag |
| `abstained` | Refused |
| `hallucinated` | Shown **without** a flag and wrong, unsupported, or answering an unanswerable question |

`hallucination_reduction = (naive_rate - grounded_rate) / naive_rate`

## Extending

- **LLM entailment check** for sentences between 40-60% support (would reduce false rejections of paraphrases)
- **Fetch full web pages** for the fallback instead of snippets
- **Feedback loop**: a thumbs-down in the app writes the question to a review queue and the eval set
- **CRM hand-off**: post the cited answer as a note on the HubSpot/Zoho deal the rep is working on
