"""Settings from config.yaml, with defaults, plus environment overrides from .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import yaml

try:  # .env is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    kb_path: str = "data/kb"
    index_path: str = "outputs/index.json"
    audit_path: str = "outputs/audit.jsonl"
    glossary_path: str = "data/glossary.yaml"
    top_k: int = 4
    embeddings: str = "none"
    min_support: float = 0.5
    coverage_weight: float = 0.6
    min_coverage: float = 0.5
    high: float = 0.75
    fallback_below: float = 0.55
    abstain_below: float = 0.45
    web_enabled: bool = True
    web_provider: str = "ddgs"
    web_max_results: int = 4

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


def load_settings(path: Optional[Path] = None) -> Settings:
    path = Path(path or os.getenv("RAG_AGENT_CONFIG", "config.yaml"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw = raw or {}
    p, r = raw.get("paths", {}), raw.get("retrieval", {})
    v, c, w = raw.get("verification", {}), raw.get("confidence", {}), raw.get("web", {})
    s = Settings()
    s = s.with_overrides(
        kb_path=p.get("kb"), index_path=p.get("index"), audit_path=p.get("audit"),
        glossary_path=p.get("glossary"),
        top_k=r.get("top_k"), embeddings=r.get("embeddings"),
        min_support=v.get("min_support"),
        coverage_weight=c.get("coverage_weight"), min_coverage=c.get("min_coverage"), high=c.get("high"),
        fallback_below=c.get("fallback_below"), abstain_below=c.get("abstain_below"),
        web_enabled=w.get("enabled"), web_provider=w.get("provider"), web_max_results=w.get("max_results"),
    )
    env_web = os.getenv("RAG_AGENT_WEB_SEARCH")
    if env_web:
        s = s.with_overrides(web_provider=env_web)
    if not (s.abstain_below <= s.fallback_below <= s.high):
        raise ValueError("config.yaml: need abstain_below <= fallback_below <= high")
    return s
