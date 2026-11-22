import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # config.yaml, data/ and fixtures use repo-relative paths

from grounded_rag.agent import Agent  # noqa: E402
from grounded_rag.config import load_settings  # noqa: E402
from grounded_rag.index import Index  # noqa: E402
from grounded_rag.llm import LLM, get_llm  # noqa: E402
from grounded_rag.websearch import MockSearch  # noqa: E402


class ScriptedLLM(LLM):
    """Stands in for Gemini: returns a fixed JSON reply, chosen by whether it was given KB or web sources."""

    name = "scripted"

    def __init__(self, kb_reply: dict, web_reply: dict | None = None):
        super().__init__("test")
        self.kb_reply, self.web_reply = kb_reply, web_reply or {"answerable": False, "sentences": []}
        self.prompts: list[tuple[str, str]] = []

    def complete_text(self, system: str, user: str, json_mode: bool = False) -> str:
        self.prompts.append((system, user))
        return json.dumps(self.web_reply if "[W1]" in user else self.kb_reply)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Tests must not depend on the developer's .env (e.g. RAG_AGENT_PROVIDER=gemini)."""
    for var in ("RAG_AGENT_PROVIDER", "RAG_AGENT_WEB_SEARCH", "GEMINI_MODEL", "RAG_AGENT_CONFIG"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="session")
def index() -> Index:
    return Index.build(ROOT / "data" / "kb")


@pytest.fixture
def settings():
    return load_settings(ROOT / "config.yaml").with_overrides(audit_path="outputs/test_audit.jsonl")


@pytest.fixture
def mock_agent(index, settings) -> Agent:
    return Agent(index, get_llm("mock"), MockSearch(), settings)


@pytest.fixture
def scripted_agent(index, settings):
    def make(kb_reply, web_reply=None, search=None):
        return Agent(index, ScriptedLLM(kb_reply, web_reply), search or MockSearch(), settings)

    return make
