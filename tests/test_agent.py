import json

from grounded_rag.agent import ABSTAIN_TEXT, write_audit
from grounded_rag.websearch import NoSearch, SearchError, WebSearch


def test_mock_agent_answers_from_kb_with_citation(mock_agent):
    r = mock_agent.ask("What is the restocking fee for opened items?")
    assert r.status == "answered" and r.route == "kb"
    assert "15%" in r.answer_text() and "[1]" in r.answer_text()
    assert r.cited_sources[0].ref == "warranty_and_returns#returns-and-restocking-fee"


def test_mock_agent_falls_back_to_web(mock_agent):
    r = mock_agent.ask("When did Windows 10 reach end of support?")
    assert r.route == "web" and r.status == "answered"
    assert "October 14, 2025" in r.answer_text()
    assert [t["step"] for t in r.trace][:3] == ["retrieve", "draft+verify (kb)", "web search"]


def test_mock_agent_abstains_on_private_data(mock_agent):
    r = mock_agent.ask("What discount did we give Syarikat Maju Jaya last March?")
    assert r.status == "abstained" and r.answer_text() == ABSTAIN_TEXT


def test_hallucinated_sentence_is_removed_but_reported(scripted_agent):
    agent = scripted_agent({"answerable": True, "sentences": [
        {"text": "Opened items that are not faulty carry a 15% restocking fee.", "sources": ["S1"]},
        {"text": "VIP clients get the restocking fee waived after 30 days.", "sources": ["S1"]},
    ]})
    r = agent.ask("What is the restocking fee for opened items?")
    assert "15%" in r.answer_text() and "VIP" not in r.answer_text()
    assert len(r.removed) == 1 and "30" in r.removed[0].issues[0]


def test_model_citing_unknown_source_is_rejected(scripted_agent):
    agent = scripted_agent({"answerable": True, "sentences": [
        {"text": "Opened items carry a 15% restocking fee.", "sources": ["S7"]},
    ]})
    r = agent.ask("What is the restocking fee for opened items?")
    assert r.status == "abstained" and r.removed[0].issues


def test_not_answerable_triggers_web_fallback(scripted_agent):
    agent = scripted_agent(
        {"answerable": False, "sentences": []},
        {"answerable": True, "sentences": [
            {"text": "Windows 10 reached end of support on October 14, 2025.", "sources": ["W1"]}]},
    )
    r = agent.ask("When did Windows 10 reach end of support?")
    assert r.route == "web" and r.cited_sources[0].kind == "web"
    assert r.cited_sources[0].ref.startswith("https://")


def test_search_failure_is_traced_and_agent_abstains(scripted_agent):
    class Broken(WebSearch):
        name = "broken"

        def search(self, query, k=4):
            raise SearchError("rate limited")

    agent = scripted_agent({"answerable": False, "sentences": []}, search=Broken())
    r = agent.ask("When did Windows 10 reach end of support?")
    assert r.status == "abstained"
    assert any("rate limited" in t["detail"] for t in r.trace)


def test_web_fallback_can_be_switched_off(index, settings):
    from grounded_rag.agent import Agent
    from grounded_rag.llm import get_llm
    from grounded_rag.websearch import MockSearch

    agent = Agent(index, get_llm("mock"), MockSearch(), settings.with_overrides(web_enabled=False))
    assert isinstance(agent.search, NoSearch)
    r = agent.ask("When did Windows 10 reach end of support?")
    assert r.route != "web"


def test_naive_mode_shows_unsupported_sentences(scripted_agent):
    agent = scripted_agent({"answerable": True, "sentences": [
        {"text": "VIP clients get the restocking fee waived after 30 days.", "sources": []}]})
    naive = agent.ask("What is the restocking fee for opened items?", mode="naive")
    grounded = agent.ask("What is the restocking fee for opened items?")
    assert naive.status == "answered" and not naive.sentences[0].supported
    assert grounded.status == "abstained"


def test_prompt_contains_numbered_sources(scripted_agent):
    agent = scripted_agent({"answerable": False, "sentences": []})
    agent.ask("What is the restocking fee for opened items?")
    system, user = agent.llm.prompts[0]
    assert "ONLY the numbered sources" in system and "[S1]" in user


def test_audit_log_line(mock_agent, tmp_path):
    r = mock_agent.ask("What is the restocking fee for opened items?")
    path = tmp_path / "audit.jsonl"
    write_audit(r, path)
    write_audit(r, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["citations"] == [r.cited_sources[0].ref]


def test_result_serialises(mock_agent):
    d = mock_agent.ask("How long does delivery to Sabah take?").to_dict()
    json.dumps(d)
    assert d["citations"][0]["n"] == 1 and d["confidence"]["level"] in ("high", "medium", "low")
