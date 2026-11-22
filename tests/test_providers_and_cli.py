import json
import sys
import types

import pytest

from grounded_rag import cli, llm, websearch
from grounded_rag.evaluate import load_cases, run_eval
from grounded_rag.llm import GeminiLLM, LLMError, parse_json_object


class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.text = data, status, json.dumps(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code))


def test_parse_json_handles_fences_and_chatter():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(LLMError):
        parse_json_object("no json here")


def test_gemini_request_shape(monkeypatch):
    sent = {}

    def fake_post(url, headers, params, json, timeout):
        sent.update(url=url, headers=headers, payload=json)
        return FakeResponse({"candidates": [{"content": {"parts": [
            {"text": "thinking...", "thought": True}, {"text": '{"answerable": false, "sentences": []}'}]}}]})

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = GeminiLLM().complete_json("sys", "user")
    assert out == {"answerable": False, "sentences": []}
    assert sent["headers"]["x-goog-api-key"] == "test-key"
    assert sent["payload"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "key=" not in sent["url"]  # key goes in a header, never in the URL


def test_gemini_needs_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        GeminiLLM()


def test_wikipedia_results_are_parsed(monkeypatch):
    def fake_get(url, params, headers, timeout):
        assert "User-Agent" in headers
        return FakeResponse({"query": {"search": [
            {"title": "DuitNow", "snippet": "<span>DuitNow</span> is a payment service"}]}})

    monkeypatch.setattr(websearch.requests, "get", fake_get)
    res = websearch.WikipediaSearch().search("DuitNow")
    assert res[0].url == "https://en.wikipedia.org/wiki/DuitNow" and "<span>" not in res[0].snippet


def test_ddgs_errors_become_search_errors(monkeypatch):
    class Boom:
        def text(self, *a, **k):
            raise RuntimeError("202 Ratelimit")

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=Boom))
    with pytest.raises(websearch.SearchError):
        websearch.DuckDuckGoSearch().search("x")


def test_ddgs_results_are_mapped(monkeypatch):
    class Fine:
        def text(self, query, max_results):
            return [{"title": "T", "href": "https://x.test", "body": "snippet"}, {"title": "empty", "body": ""}]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=Fine))
    res = websearch.DuckDuckGoSearch().search("x")
    assert len(res) == 1 and res[0].url == "https://x.test"


def test_mock_search_only_matches_relevant_fixtures():
    s = websearch.MockSearch()
    assert s.search("When did Windows 10 support end?")[0].url.endswith("windows-10-end-of-support")
    assert s.search("favourite colour of the CEO") == []


def test_eval_shows_fewer_hallucinations_than_naive(mock_agent, tmp_path):
    report = run_eval(mock_agent, load_cases("data/eval/questions.jsonl"), tmp_path)
    naive = report["modes"]["naive"]["summary"]
    grounded = report["modes"]["grounded"]["summary"]
    assert grounded["hallucinated_answer_rate"] < naive["hallucinated_answer_rate"]
    assert grounded["unanswerable_abstain_or_flag_rate"] == 1.0
    assert (tmp_path / "eval_report.md").exists()


def test_cli_ingest_and_ask(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"paths:\n  kb: data/kb\n  index: {tmp_path.as_posix()}/index.json\n"
                   f"  audit: {tmp_path.as_posix()}/audit.jsonl\nweb:\n  provider: mock\n", encoding="utf-8")
    assert cli.main(["--config", str(cfg), "ingest"]) == 0
    assert cli.main(["--config", str(cfg), "ask", "What", "is", "the", "rush", "delivery", "fee?", "--json"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out[out.index("{"):])
    assert "150" in data["answer"] and data["citations"]
    assert (tmp_path / "audit.jsonl").exists()


def test_cli_reports_missing_key_cleanly(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert cli.main(["ask", "hello", "--provider", "gemini"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


class ErrorResponse(FakeResponse):
    def __init__(self, status, body):
        super().__init__(body, status)
        self.headers = {}


def test_rate_limit_waits_as_asked_then_succeeds(monkeypatch):
    calls, waits = [], []
    replies = [
        ErrorResponse(429, {"error": {"message": "per minute", "details": [{"retryDelay": "7s"}]}}),
        FakeResponse({"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}),
    ]

    def fake_post(*a, **k):
        calls.append(1)
        return replies.pop(0)

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    assert GeminiLLM().complete_json("s", "u") == {"ok": True}
    assert waits == [8.0] and len(calls) == 2


def test_daily_quota_stops_immediately(monkeypatch):
    def fake_post(*a, **k):
        return ErrorResponse(429, {"error": {"message": "Quota exceeded",
                                             "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}})

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: pytest.fail("should not wait for a daily quota"))
    with pytest.raises(llm.QuotaExhausted):
        GeminiLLM().complete_json("s", "u")


def test_model_errors_are_not_counted_as_refusals(index, settings, tmp_path):
    from grounded_rag.agent import Agent
    from grounded_rag.evaluate import EvalCase, is_trustworthy
    from grounded_rag.llm import LLM
    from grounded_rag.websearch import NoSearch

    class Flaky(LLM):
        name = "flaky"

        def complete_text(self, system, user, json_mode=False):
            raise LLMError("Rate limited (429) after 4 retries")

    agent = Agent(index, Flaky("x"), NoSearch(), settings)
    r = agent.ask("What is the restocking fee for opened items?")
    assert r.status == "abstained" and r.errors

    cases = [EvalCase("kb02", "kb", "What is the restocking fee for opened items?", ["15%"], ["warranty_and_returns"])]
    report = run_eval(agent, cases, tmp_path)
    g = report["modes"]["grounded"]["summary"]
    assert g["errors"] == 1 and g["questions"] == 0 and g["abstain_rate"] is None
    assert not is_trustworthy(report)
    assert "hit model errors" in (tmp_path / "eval_report.md").read_text(encoding="utf-8")


def test_eval_stops_cleanly_when_daily_quota_runs_out(index, settings, tmp_path):
    from grounded_rag.agent import Agent
    from grounded_rag.llm import LLM
    from grounded_rag.websearch import NoSearch

    class OutOfQuota(LLM):
        name = "out"

        def complete_text(self, system, user, json_mode=False):
            raise llm.QuotaExhausted("Daily free-tier quota for this model is used up.")

    agent = Agent(index, OutOfQuota("x"), NoSearch(), settings)
    msgs = []
    report = run_eval(agent, load_cases("data/eval/questions.jsonl"), tmp_path, progress=msgs.append)
    assert report["stopped_early"].startswith("after 0 of 25")
    assert msgs and msgs[-1].startswith("STOPPED")
