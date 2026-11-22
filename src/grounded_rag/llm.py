"""Swappable LLM providers.

All providers expose the same two methods, so the rest of the code never
knows which model it is talking to:

    complete_text(system, user) -> str
    complete_json(system, user) -> dict

Supported: Gemini (free tier), Claude, Ollama (local, free) and "mock"
(offline, no model at all - used for tests, CI and zero-cost demos).
Only `requests` is used, so there are no vendor SDKs to install.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import requests

DEFAULT_MODELS = {
    "gemini": "gemini-3.6-flash",
    "claude": "claude-haiku-5-5",
    "ollama": "llama3.1",
}

TIMEOUT_SECONDS = int(os.getenv("RAG_AGENT_LLM_TIMEOUT", "180"))


class LLMError(RuntimeError):
    pass


class LLM:
    name = "base"

    def __init__(self, model: Optional[str] = None):
        self.model = model or DEFAULT_MODELS.get(self.name, "")

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}" if self.model else self.name

    def complete_text(self, system: str, user: str, json_mode: bool = False) -> str:  # pragma: no cover
        raise NotImplementedError

    def complete_json(self, system: str, user: str) -> dict:
        raw = self.complete_text(system, user, json_mode=True)
        return parse_json_object(raw)


def parse_json_object(raw: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences or chatter."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"Model did not return JSON: {raw[:200]!r}")
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise LLMError("Model returned JSON but not an object")
    return data


class QuotaExhausted(LLMError):
    """The provider's daily free-tier quota is used up: retrying today won't help."""


MAX_RATE_LIMIT_WAIT = 65  # seconds; Gemini's per-minute limits reset within a minute


def _retry_after(resp) -> Optional[float]:
    """Seconds the provider asks us to wait: Retry-After header, or Gemini's RetryInfo.retryDelay ("37s")."""
    header = (getattr(resp, "headers", None) or {}).get("Retry-After")
    try:
        return float(header)
    except (TypeError, ValueError):
        pass
    try:
        for detail in resp.json().get("error", {}).get("details", []):
            if detail.get("retryDelay"):
                return float(str(detail["retryDelay"]).rstrip("s"))
    except (ValueError, AttributeError):
        pass
    return None


def _is_daily_quota(text: str) -> bool:
    compact = text.lower().replace(" ", "").replace("_", "")
    return "perday" in compact


def _post_with_retry(url: str, *, headers: dict, payload: dict, params: Optional[dict] = None) -> dict:
    """POST with backoff on rate limits (429), transient 5xx errors and timeouts.

    On 429 it waits as long as the provider asks (up to MAX_RATE_LIMIT_WAIT). If the 429 says the
    *daily* quota is used up, it stops at once with QuotaExhausted instead of retrying for minutes.
    """
    server_delays = [2, 5, 10]
    rate_limit_tries = 4
    attempt = 0
    while True:
        try:
            resp = requests.post(url, headers=headers, params=params, json=payload, timeout=TIMEOUT_SECONDS)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt < 1:  # one retry for network problems, then give a clear error
                attempt += 1
                time.sleep(server_delays[0])
                continue
            raise LLMError(
                f"No answer from {url.split('/v1')[0]} within {TIMEOUT_SECONDS}s ({type(exc).__name__}). "
                "Check your internet connection, try again, or set RAG_AGENT_LLM_TIMEOUT higher in .env."
            ) from exc

        if resp.status_code == 429:
            if _is_daily_quota(resp.text):
                raise QuotaExhausted(
                    "Daily free-tier quota for this model is used up. Wait for it to reset (midnight US Pacific "
                    "time, about 3-4 pm in Malaysia), or set GEMINI_MODEL in .env to another model "
                    "(python scripts/check_gemini.py --list)."
                )
            if attempt < rate_limit_tries:
                wait = _retry_after(resp)
                wait = min(MAX_RATE_LIMIT_WAIT, wait + 1 if wait is not None else 10 * (attempt + 1))
                attempt += 1
                time.sleep(wait)
                continue
            raise LLMError(f"Rate limited (429) after {rate_limit_tries} retries: {resp.text[:200]}")

        if resp.status_code in (500, 502, 503, 529) and attempt < len(server_delays):
            time.sleep(server_delays[attempt])
            attempt += 1
            continue
        if resp.status_code >= 400:
            raise LLMError(f"{url} returned {resp.status_code}: {resp.text[:300]}")
        return resp.json()


class GeminiLLM(LLM):
    """Google Gemini via the public REST API. Free tier: get a key at aistudio.google.com."""

    name = "gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(model or os.getenv("GEMINI_MODEL"))
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set (see .env.example)")
        # low / medium / high (some models also accept minimal). Empty = model default.
        self.thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low").strip() or None

    def complete_text(self, system: str, user: str, json_mode: bool = False) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        config: dict = {"temperature": 0}
        if json_mode:
            config["responseMimeType"] = "application/json"
        if self.thinking_level:
            # Newer Gemini models "think" before answering. Answering from given sources is simple, so keep
            # thinking short: much faster, and fewer free-tier tokens used.
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": config,
        }
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        try:
            data = _post_with_retry(url, headers=headers, payload=payload)
        except LLMError as exc:
            # Some models don't accept a thinking setting: retry once without it.
            if "thinkingConfig" in config and "returned 400" in str(exc):
                config.pop("thinkingConfig")
                self.thinking_level = None
                data = _post_with_retry(url, headers=headers, payload=payload)
            else:
                raise
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Gemini response: {json.dumps(data)[:300]}") from exc


class ClaudeLLM(LLM):
    """Anthropic Claude via the Messages API. Key from platform.claude.com."""

    name = "claude"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(model or os.getenv("CLAUDE_MODEL"))
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set (see .env.example)")

    def complete_text(self, system: str, user: str, json_mode: bool = False) -> str:
        if json_mode:
            system += "\n\nRespond with a single JSON object only. No prose, no markdown fences."
        payload = {
            "model": self.model,
            "max_tokens": 2000,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = _post_with_retry(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            payload=payload,
        )
        try:
            return "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
        except KeyError as exc:
            raise LLMError(f"Unexpected Claude response: {json.dumps(data)[:300]}") from exc


class OllamaLLM(LLM):
    """A local model through Ollama (ollama.com). Free and private, needs a decent laptop."""

    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        super().__init__(model or os.getenv("OLLAMA_MODEL"))
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def complete_text(self, system: str, user: str, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            payload["format"] = "json"
        data = _post_with_retry(f"{self.host}/api/chat", headers={}, payload=payload)
        return data.get("message", {}).get("content", "")


class MockLLM(LLM):
    """No model. The answer step switches to an offline extractive answerer instead
    (it copies the best-matching source sentences), so tests and demos are free."""

    name = "mock"

    def complete_text(self, system: str, user: str, json_mode: bool = False) -> str:
        raise LLMError("MockLLM does not generate text; the extractive answerer is used instead")


PROVIDERS = {"gemini": GeminiLLM, "claude": ClaudeLLM, "ollama": OllamaLLM, "mock": MockLLM}


def default_provider_name() -> str:
    explicit = os.getenv("RAG_AGENT_PROVIDER")
    if explicit:
        return explicit
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    return "mock"


def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> LLM:
    provider = (provider or default_provider_name()).lower()
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider {provider!r}. Choose from: {', '.join(PROVIDERS)}")
    return PROVIDERS[provider](model=model)
