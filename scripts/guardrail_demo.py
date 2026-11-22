"""Show the citation check catching a hallucination, with no API key.

A scripted "model" answers a warranty question with one true sentence and two
invented ones (a wrong price and a promise that isn't in any policy). The agent
keeps the true sentence and removes the other two, saying why.

    python scripts/guardrail_demo.py            # prints the result
    python scripts/guardrail_demo.py --save     # also writes examples/guardrail_demo.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grounded_rag.agent import Agent  # noqa: E402
from grounded_rag.config import load_settings  # noqa: E402
from grounded_rag.index import Index  # noqa: E402
from grounded_rag.llm import LLM  # noqa: E402
from grounded_rag.websearch import NoSearch  # noqa: E402

QUESTION = "How much does the extended warranty cost and what does it include?"
MODEL_REPLY = {
    "answerable": True,
    "sentences": [
        {"text": "The extended warranty costs 8% of the device's selling price and brings total cover "
                 "to 36 months for laptops and desktops.", "sources": ["S1"]},
        {"text": "It also includes free accidental damage cover for 24 months.", "sources": ["S1"]},
        {"text": "The extended warranty costs 5% for orders above MYR 50,000.", "sources": ["S1"]},
    ],
}


class OverconfidentModel(LLM):
    name = "scripted-demo"

    def complete_text(self, system: str, user: str, json_mode: bool = False) -> str:
        return json.dumps(MODEL_REPLY)


def main() -> None:
    import os

    os.chdir(ROOT)
    settings = load_settings(ROOT / "config.yaml")
    index = Index.build(ROOT / settings.kb_path)
    agent = Agent(index, OverconfidentModel("demo"), NoSearch(), settings)
    result = agent.ask(QUESTION)

    lines = ["# Guardrail demo: the citation check catching a hallucination", "",
             "A scripted model drafted three sentences. Only the first one is in the policy.", "",
             "**What the model drafted:**", ""]
    lines += [f"- {s['text']} `{s['sources']}`" for s in MODEL_REPLY["sentences"]]
    lines += ["", "**What the user sees:**", "", result.to_markdown()]
    text = "\n".join(lines)
    print(text)
    if "--save" in sys.argv:
        out = ROOT / "examples" / "guardrail_demo.md"
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
