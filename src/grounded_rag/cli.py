"""Command line: rag-agent ingest | ask | eval | demo

Examples
  rag-agent ingest                                  build the index from data/kb
  rag-agent ask "What is the restocking fee?"        answer with citations
  rag-agent ask "When did Windows 10 support end?" --web ddgs
  rag-agent eval --provider gemini                  naive vs grounded, writes outputs/eval/
  rag-agent demo                                    offline sample run, writes examples/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent, write_audit
from .config import load_settings
from .evaluate import is_trustworthy, load_cases, run_eval
from .index import Index
from .llm import LLMError, get_llm
from .websearch import SearchError, get_search

DEMO_QUESTIONS = [
    "What is the restocking fee for opened items?",
    "When should the M365 renewal quote be sent?",
    "Which laptop should we recommend for a graphic designer?",
    "When did Windows 10 reach end of support?",
    "What discount did we give Syarikat Maju Jaya last March?",
]


def _agent(args) -> Agent:
    settings = load_settings(args.config).with_overrides(top_k=args.top_k)
    if args.no_web:
        settings = settings.with_overrides(web_enabled=False)
    web = args.web or settings.web_provider
    settings = settings.with_overrides(web_provider=web)
    index = Index.load_or_build(Path(settings.index_path), Path(settings.kb_path), settings.embeddings)
    return Agent(index, get_llm(args.provider, args.model), get_search(web), settings)


def cmd_ingest(args) -> int:
    settings = load_settings(args.config)
    embeddings = args.embeddings or settings.embeddings
    index = Index.build(Path(args.kb or settings.kb_path), embeddings)
    index.save(Path(settings.index_path))
    docs = len({c.doc_id for c in index.chunks})
    print(f"Indexed {len(index.chunks)} sections from {docs} documents "
          f"(search: {'BM25 + ' + embeddings + ' embeddings' if index.vectors else 'BM25 keyword'}) "
          f"-> {settings.index_path}")
    return 0


def cmd_ask(args) -> int:
    agent = _agent(args)
    result = agent.ask(" ".join(args.question))
    write_audit(result, Path(agent.settings.audit_path))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if not result.errors else 3
    else:
        print(result.to_markdown())
        if args.trace:
            print("\nTrace:")
            for t in result.trace:
                print(f"  {t['step']:<20} {t['ms']:>6} ms  {t['detail']}")
    return 0 if not result.errors else 3


def cmd_eval(args) -> int:
    agent = _agent(args)
    cases = load_cases(Path(args.questions))
    delay = args.delay if args.delay is not None else (0.0 if agent.llm.name == "mock" else 4.0)
    print(f"Evaluating {len(cases)} questions x 2 pipelines with {agent.llm.label} "
          f"(web: {agent.search.name}, {delay:g}s pause between calls)...", flush=True)
    report = run_eval(agent, cases, Path(args.out), delay=delay, progress=lambda m: print(m, flush=True))
    md = Path(args.out) / "eval_report.md"
    print()
    print(md.read_text(encoding="utf-8").split("## Per question")[0].strip())
    print(f"\nFull report: {md}")
    return 0 if is_trustworthy(report) else 3


def cmd_demo(args) -> int:
    args.provider, args.web = args.provider or "mock", args.web or "mock"
    agent = _agent(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = [f"# Sample answers ({agent.llm.label}, web search: {agent.search.name})", ""]
    for q in DEMO_QUESTIONS:
        r = agent.ask(q)
        parts += [r.to_markdown(), "", "---", ""]
        print(f"[{r.status:<14}] conf {r.confidence.value:.2f} via {r.route:<4} | {q}")
    (out / "sample_answers.md").write_text("\n".join(parts), encoding="utf-8")
    run_eval(agent, load_cases(Path(args.questions)), out)
    print(f"Wrote {out / 'sample_answers.md'} and {out / 'eval_report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-agent", description="Grounded RAG agent with citations and web fallback")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--provider", choices=["gemini", "claude", "ollama", "mock"], default=None,
                        help="LLM (default: gemini if GEMINI_API_KEY is set, else mock)")
        sp.add_argument("--model", default=None)
        sp.add_argument("--web", choices=["ddgs", "wikipedia", "tavily", "mock", "none"], default=None)
        sp.add_argument("--no-web", action="store_true", help="switch off the web fallback")
        sp.add_argument("--top-k", type=int, default=None)

    sp = sub.add_parser("ingest", help="build the search index from the knowledge base")
    sp.add_argument("--kb", default=None, help="folder of .md/.txt/.pdf files (default from config.yaml)")
    sp.add_argument("--embeddings", choices=["none", "gemini"], default=None)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("ask", help="ask a question")
    sp.add_argument("question", nargs="+")
    sp.add_argument("--json", action="store_true", help="print the full result as JSON")
    sp.add_argument("--trace", action="store_true", help="show each step and how long it took")
    common(sp)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("eval", help="compare naive RAG with the grounded agent")
    sp.add_argument("--questions", default="data/eval/questions.jsonl")
    sp.add_argument("--out", default="outputs/eval")
    sp.add_argument("--delay", type=float, default=None,
                    help="seconds to pause after each question/pipeline (default 4 for real models, "
                         "keeps the Gemini free tier under its per-minute limit)")
    common(sp)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("demo", help="offline demo run (mock model + fixture web search)")
    sp.add_argument("--questions", default="data/eval/questions.jsonl")
    sp.add_argument("--out", default="examples")
    common(sp)
    sp.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for attr in ("provider", "model", "web", "no_web", "top_k"):
        if not hasattr(args, attr):
            setattr(args, attr, None)
    try:
        return args.func(args)
    except (LLMError, SearchError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
