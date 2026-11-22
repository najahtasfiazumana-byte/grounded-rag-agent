# Demo script (about 3 minutes)

For a screen recording or a live interview demo. Have `streamlit run app.py` open with Gemini
selected, plus a terminal.

## 0:00 The problem (20 s)

> "Sales reps ask the same policy questions all day: restocking fees, who approves a discount, when to
> send a renewal quote. A plain chatbot answers confidently even when it's making things up, and in
> sales a made-up discount rule or fee costs real money. So I built a RAG agent that has to prove every
> sentence."

## 0:20 A normal question (40 s)

Click **"What is the restocking fee for opened items?"**

> "It searches the sales playbook, and the model drafts an answer where each sentence names its source.
> Then code, not the model, checks every sentence: does the citation exist, is every number in the
> source, does the wording match? Here it's 15%, cited to the exact policy section. I can open the
> source and check it in five seconds."

Point at **How it got here**: retrieve, draft and verify, decide, with timings.

## 1:00 Not in the documents: web fallback (30 s)

Click **"When did Windows 10 reach end of support?"**

> "That isn't in our playbook. Confidence came out low, so it fell back to live web search, ran the same
> citation check on the results, and now cites a URL instead of a policy."

## 1:30 Nobody knows: refusal (20 s)

Click **"What discount did we give Syarikat Maju Jaya last March?"**

> "That's private data no document contains. A naive bot would grab something about discounts and
> answer. This one says it won't guess."

## 1:50 Catching a hallucination (30 s)

In the terminal: `python scripts/guardrail_demo.py`

> "Here a model drafted three sentences about the extended warranty. Two were invented: accidental damage
> cover and a 5% bulk rate. The number check and wording check removed both and show why, so a reviewer
> sees what the model tried to slip in."

## 2:20 Proof: the eval (30 s)

Open the **Evaluation** tab.

> "I didn't want to just claim it reduces hallucinations, so I built an eval: 25 questions, including
> ones only on the web and ones nobody should answer. Same model, naive RAG versus my agent. Hallucinated
> answers went from __% to __%, and every unanswerable question was refused or flagged."

*(Fill in the numbers from your own `rag-agent eval --provider gemini` run.)*

## 2:50 Close (10 s)

> "It's all on free tiers (Gemini, DuckDuckGo, BM25 in plain Python), has 47 tests running in GitHub
> Actions, and you can point it at any folder of documents."

---

## Likely interview questions

**Why not just use a vector database?** For a few hundred policy sections, BM25 is free, exact on terms
like "Net 30" and product names, and runs in CI with no key. Embeddings are an option, merged with
Reciprocal Rank Fusion. I'd add a vector store when the corpus reaches tens of thousands of documents.

**Why not ask the model how confident it is?** LLM self-confidence is poorly calibrated. My score is
computed from things I can test: how much of the question the cited source covers, and how many sentences
passed verification.

**What are the weaknesses?** The checks are lexical, so a heavy paraphrase can be rejected and a sentence
that only mentions a term can pass. Next I'd add an entailment check for borderline sentences. Coverage is
deliberately strict, so multi-document questions get flagged instead of answered.

**How would this fit into a GTM stack?** As a Slack or CRM side panel for reps, with the cited answer
posted as a note on the HubSpot or Zoho deal. Questions it refuses become a list of gaps in the playbook
for sales ops to document.
