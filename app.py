"""Streamlit UI for the grounded RAG agent.

Run:  streamlit run app.py
Works with no API key (offline mock model); add GEMINI_API_KEY to .env for real answers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from grounded_rag.agent import Agent, write_audit
from grounded_rag.config import load_settings
from grounded_rag.evaluate import METRICS, is_trustworthy, load_cases, run_eval
from grounded_rag.index import Index
from grounded_rag.llm import LLMError, get_llm
from grounded_rag.websearch import SearchError, get_search

st.set_page_config(page_title="Grounded RAG Agent", page_icon="📚", layout="wide")

SETTINGS = load_settings()
EXAMPLES = [
    "What is the restocking fee for opened items?",
    "When should the M365 renewal quote be sent?",
    "Which laptop should we recommend for a graphic designer?",
    "Who needs to approve a 12% discount?",
    "When did Windows 10 reach end of support?",
    "What discount did we give Syarikat Maju Jaya last March?",
]
SERIES = {"Naive RAG": "#2a78d6", "Grounded agent": "#eb6834"}  # fixed categorical order


@st.cache_resource(show_spinner="Building the search index...")
def get_index(kb_path: str, index_path: str, embeddings: str, _bust: int = 0) -> Index:
    return Index.load_or_build(Path(index_path), Path(kb_path), embeddings)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    providers = ["gemini", "mock"] if has_gemini else ["mock", "gemini"]
    provider = st.radio("Model", providers,
                        format_func=lambda p: {"gemini": "Gemini (free tier)", "mock": "Offline (no key)"}[p],
                        help="Offline mode copies the best-matching source sentences, so it runs with no key.")
    if provider == "gemini" and not has_gemini:
        st.warning("Add GEMINI_API_KEY to your .env file, then restart.")
    web_options = ["ddgs", "wikipedia", "tavily", "mock"]
    web_provider = st.selectbox(
        "Live search fallback", web_options,
        index=web_options.index(SETTINGS.web_provider) if SETTINGS.web_provider in web_options else 0,
        format_func=lambda w: {"ddgs": "DuckDuckGo (no key)", "wikipedia": "Wikipedia (no key)",
                               "tavily": "Tavily (free key)", "mock": "Offline fixtures"}[w])
    web_on = st.toggle("Use web search when confidence is low", value=SETTINGS.web_enabled)
    top_k = st.slider("Sections given to the model", 2, 8, SETTINGS.top_k)
    st.caption(
        f"Fallback below **{SETTINGS.fallback_below:.2f}** confidence · refuse below "
        f"**{SETTINGS.abstain_below:.2f}** · min. evidence per sentence **{SETTINGS.min_support:.0%}** "
        "(edit in config.yaml)")
    if st.button("Rebuild index"):
        st.session_state["bust"] = st.session_state.get("bust", 0) + 1
        Path(SETTINGS.index_path).unlink(missing_ok=True)

index = get_index(SETTINGS.kb_path, SETTINGS.index_path, SETTINGS.embeddings, st.session_state.get("bust", 0))
settings = SETTINGS.with_overrides(top_k=top_k, web_enabled=web_on, web_provider=web_provider)


def make_agent() -> Agent | None:
    try:
        return Agent(index, get_llm(provider), get_search(web_provider), settings)
    except (LLMError, SearchError) as exc:
        st.error(str(exc))
        return None


st.title("📚 Grounded RAG Agent")
st.caption("Answers from the sales playbook with a citation on every sentence. Each sentence is checked "
           "against its source, low-confidence answers are flagged, and live web search is the fallback.")

tab_ask, tab_kb, tab_eval = st.tabs(["Ask", "Knowledge base", "Evaluation"])

# ----------------------------------------------------------------------------- ask
with tab_ask:
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state["question"] = ex
    question = st.text_input("Ask a question", key="question",
                             placeholder="e.g. How long does delivery to Sabah take?")

    if question:
        agent = make_agent()
        if agent is not None:
            with st.spinner("Searching, drafting and checking citations..."):
                try:
                    result = agent.ask(question)
                except LLMError as exc:
                    st.error(f"Model error: {exc}")
                    result = None
            if result is not None:
                write_audit(result, Path(settings.audit_path))
                if result.errors:
                    st.error("The model call failed, so the result below is not a real refusal: "
                             + " | ".join(result.errors))
                conf = result.confidence
                route = {"kb": "knowledge base", "web": "live web search", "none": "no source"}[result.route]
                summary = f"Confidence {conf.value:.2f} ({conf.level}) · source: {route}"
                if result.status == "answered":
                    st.success(f"✅ Answered · {summary}")
                elif result.status == "low_confidence":
                    st.warning(f"⚠️ Low confidence: verify before sending to a client · {summary}")
                else:
                    st.error(f"🛑 Not answered: the agent would rather not guess · {summary}")

                st.markdown(f"#### {result.answer_text()}")
                st.progress(min(max(conf.value, 0.0), 1.0))
                st.caption("Why: " + "; ".join(conf.reasons) +
                           f" · question coverage {conf.coverage:.0%} · verified sentences {conf.grounded_share:.0%}")

                left, right = st.columns([3, 2])
                with left:
                    st.subheader("Sources")
                    if not result.cited_sources:
                        st.write("No source was strong enough to cite.")
                    for n, src in enumerate(result.cited_sources, start=1):
                        with st.expander(f"[{n}] {src.label}", expanded=n == 1):
                            if src.kind == "web":
                                st.markdown(f"🌐 [{src.ref}]({src.ref})")
                            else:
                                st.code(src.ref, language=None)
                            st.write(src.text)
                    if result.removed:
                        with st.expander(f"🧹 {len(result.removed)} sentence(s) removed by the citation check"):
                            for r in result.removed:
                                st.markdown(f"~~{r.text}~~  \n*{'; '.join(r.issues)}*")
                with right:
                    st.subheader("How it got here")
                    st.dataframe(pd.DataFrame(result.trace), hide_index=True, width="stretch")
                    uncited = [s for s in result.sources if s not in result.cited_sources]
                    if uncited:
                        with st.expander("Other sections retrieved but not cited"):
                            for s in uncited:
                                st.markdown(f"**{s.id}** {s.label}")
                    st.download_button("Download result (JSON)", json.dumps(result.to_dict(), indent=2),
                                       file_name="answer.json", mime="application/json")

# ----------------------------------------------------------------------------- knowledge base
with tab_kb:
    titles = {c.doc_id: c.doc_title for c in index.chunks}
    doc_ids = sorted(titles, key=titles.get)
    st.write(f"**{len(index.chunks)}** sections from **{len(doc_ids)}** documents in `{settings.kb_path}` "
             f"· search: {'BM25 + embeddings' if index.vectors else 'BM25 keyword'} · built {index.built_at}")
    choice = st.selectbox("Document", doc_ids, format_func=titles.get)
    for c in [c for c in index.chunks if c.doc_id == choice]:
        with st.expander(c.section or "(intro)"):
            st.code(c.id, language=None)
            st.write(c.text)

# ----------------------------------------------------------------------------- evaluation
with tab_eval:
    st.write("Same questions and model, two pipelines: **naive RAG** (retrieve, answer, show everything) vs the "
             "**grounded agent** (citation check, confidence gate, web fallback, refusal).")
    out_dir = Path("outputs/eval")
    candidates = [p for p in (out_dir / "eval_report.json", Path("examples/eval_report.json")) if p.exists()]
    if st.button("Run evaluation now (25 questions × 2 pipelines)"):
        agent = make_agent()
        if agent is not None:
            if provider == "gemini":
                st.info("This makes about 50-60 Gemini calls. On the free tier it can take a few minutes.")
            with st.spinner("Running... (progress is printed in the terminal)"):
                run_eval(agent, load_cases(Path("data/eval/questions.jsonl")), out_dir,
                         delay=0.0 if provider == "mock" else 4.0, progress=print)
            candidates = [out_dir / "eval_report.json"]
    if not candidates:
        st.info("No evaluation yet. Click the button above or run `rag-agent eval`.")
    else:
        report = json.loads(candidates[0].read_text(encoding="utf-8"))
        n, g = report["modes"]["naive"]["summary"], report["modes"]["grounded"]["summary"]
        st.caption(f"Report `{candidates[0]}` · model `{report['provider']}` · web `{report['web_provider']}` · "
                   f"{report['run_at']}")
        if report["provider"].startswith("mock"):
            st.warning("Offline mock model: these numbers show the pipeline works. Run with Gemini for real numbers.")
        if not is_trustworthy(report):
            st.error(f"Incomplete run: {n.get('errors', 0)} question(s) hit model errors"
                     + (f"; stopped early {report['stopped_early']}" if report.get("stopped_early") else "")
                     + ". Those questions are left out of the numbers below. Re-run before quoting them.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Hallucinated answers, naive", f"{n['hallucinated_answer_rate']:.0%}")
        c2.metric("Hallucinated answers, grounded", f"{g['hallucinated_answer_rate']:.0%}")
        red = report.get("hallucination_reduction")
        c3.metric("Reduction", "n/a" if red is None else f"{red:.0%}")

        rows = []
        for key, label in METRICS:
            for mode, name in (("naive", "Naive RAG"), ("grounded", "Grounded agent")):
                v = report["modes"][mode]["summary"][key]
                if v is not None:
                    rows.append({"Metric": label, "Pipeline": name, "Rate": v})
        df = pd.DataFrame(rows)
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=12)
            .encode(
                y=alt.Y("Metric:N", sort=[m[1] for m in METRICS], title=None, axis=alt.Axis(labelLimit=320)),
                yOffset=alt.YOffset("Pipeline:N", sort=list(SERIES)),
                x=alt.X("Rate:Q", axis=alt.Axis(format="%", grid=True, gridOpacity=0.3), title=None,
                        scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Pipeline:N", scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values())),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["Pipeline", "Metric", alt.Tooltip("Rate:Q", format=".0%")],
            )
            .properties(height=len(METRICS) * 38)
        )
        st.altair_chart(chart, width="stretch")
        table = df.pivot(index="Metric", columns="Pipeline", values="Rate").reindex([m[1] for m in METRICS])
        st.dataframe(table.style.format("{:.0%}"), width="stretch")
        with st.expander("Per question (grounded agent)"):
            st.dataframe(pd.DataFrame(report["modes"]["grounded"]["rows"])
                         [["id", "type", "question", "status", "route", "confidence", "outcome"]],
                         hide_index=True, width="stretch")
