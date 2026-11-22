"""Live web search, used only as a fallback when the knowledge base can't answer.

Providers (all free):
  ddgs       DuckDuckGo results through the `ddgs` package. No key. Default.
  wikipedia  Wikipedia's public search API. No key. Good for general facts.
  tavily     Tavily search API built for AI agents. Free key (1,000 searches/month).
  mock       Offline fixtures from data/web_fixtures.json, for tests, CI and demos.
  none       Fallback switched off.

Web results become sources W1, W2, ... and go through exactly the same
citation check as knowledge-base sources.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests

from .text import content_terms

USER_AGENT = "grounded-rag-agent/0.1 (portfolio project; https://github.com/najahtasfiazumana-byte/grounded-rag-agent)"
TIMEOUT = 20


class SearchError(RuntimeError):
    pass


def _short(exc: Exception, limit: int = 160) -> str:
    """First line of an error, trimmed: network errors can be several hundred characters long."""
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:limit]}{'...' if len(text) > limit else ''}"


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


class WebSearch:
    name = "base"

    def search(self, query: str, k: int = 4) -> list[WebResult]:  # pragma: no cover
        raise NotImplementedError


class DuckDuckGoSearch(WebSearch):
    name = "ddgs"

    def search(self, query: str, k: int = 4) -> list[WebResult]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise SearchError("The ddgs package is not installed: pip install ddgs") from exc
        try:
            rows = DDGS().text(query, max_results=k) or []
        except Exception as exc:  # ddgs raises its own exception types for rate limits/timeouts
            raise SearchError(f"DuckDuckGo search failed: {_short(exc)}") from exc
        return [WebResult(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in rows if r.get("body")]


class WikipediaSearch(WebSearch):
    name = "wikipedia"

    def search(self, query: str, k: int = 4) -> list[WebResult]:
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query, "srlimit": k,
                        "format": "json", "utf8": 1},
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SearchError(f"Wikipedia search failed: {_short(exc)}") from exc
        results = []
        for row in resp.json().get("query", {}).get("search", []):
            snippet = re.sub(r"<[^>]+>", "", row.get("snippet", ""))
            title = row.get("title", "")
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            results.append(WebResult(title, url, snippet))
        return results


class TavilySearch(WebSearch):
    name = "tavily"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise SearchError("TAVILY_API_KEY is not set (free key at tavily.com)")

    def search(self, query: str, k: int = 4) -> list[WebResult]:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"query": query, "max_results": k, "search_depth": "basic"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SearchError(f"Tavily search failed: {_short(exc)}") from exc
        return [WebResult(r.get("title", ""), r.get("url", ""), r.get("content", ""))
                for r in resp.json().get("results", [])]


class MockSearch(WebSearch):
    """Returns canned results whose keywords overlap the query. Clearly labelled as fixtures."""

    name = "mock"

    def __init__(self, fixtures_path: Optional[Path] = None):
        path = Path(fixtures_path or os.getenv("RAG_AGENT_WEB_FIXTURES", "data/web_fixtures.json"))
        self.fixtures = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def search(self, query: str, k: int = 4) -> list[WebResult]:
        q = content_terms(query)
        scored = []
        for fx in self.fixtures:
            keys = content_terms(" ".join(fx["keywords"]))
            overlap = len(q & keys)
            if overlap >= max(1, min(2, len(keys))):
                scored.append((overlap, fx))
        scored.sort(key=lambda s: -s[0])
        return [WebResult(r["title"], r["url"], r["snippet"]) for _, fx in scored for r in fx["results"]][:k]


class NoSearch(WebSearch):
    name = "none"

    def search(self, query: str, k: int = 4) -> list[WebResult]:
        return []


PROVIDERS = {
    "ddgs": DuckDuckGoSearch,
    "wikipedia": WikipediaSearch,
    "tavily": TavilySearch,
    "mock": MockSearch,
    "none": NoSearch,
}


def get_search(name: Optional[str] = None) -> WebSearch:
    name = (name or os.getenv("RAG_AGENT_WEB_SEARCH") or "ddgs").lower()
    if name not in PROVIDERS:
        raise SearchError(f"Unknown web search provider {name!r}. Choose from: {', '.join(PROVIDERS)}")
    return PROVIDERS[name]()
