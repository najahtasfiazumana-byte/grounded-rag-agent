from grounded_rag.answer import Draft
from grounded_rag.confidence import score
from grounded_rag.sources import Source
from grounded_rag.verify import check_sentence, verify_draft

SRC = {
    "S1": Source("S1", "kb", "Returns", "w#returns",
                 "Opened items that are not faulty can be returned within 14 days of delivery "
                 "but carry a 15% restocking fee."),
    "S2": Source("S2", "kb", "Delivery", "d#times", "Deliveries to Sabah and Sarawak take 7 to 10 business days."),
}


def test_supported_sentence_passes():
    c = check_sentence("Opened items carry a 15% restocking fee if returned within 14 days.", ["S1"], SRC, 0.5)
    assert c.supported and not c.issues


def test_invented_number_is_caught():
    c = check_sentence("Opened items carry a 20% restocking fee.", ["S1"], SRC, 0.5)
    assert not c.supported and "20%" in c.issues[0]


def test_percentage_and_plain_number_are_different_facts():
    src = {"S1": Source("S1", "kb", "x", "x", "The fee is MYR 15 per device.")}
    c = check_sentence("The fee is 15% per device.", ["S1"], src, 0.5)
    assert not c.supported


def test_number_formatting_does_not_matter():
    src = {"S1": Source("S1", "kb", "x", "x", "Delivery is free for orders of MYR 3,000 and above.")}
    assert check_sentence("Orders of MYR 3000 or more ship free.", ["S1"], src, 0.5).supported


def test_missing_and_made_up_citations_fail():
    assert check_sentence("Returns are free.", [], SRC, 0.5).issues == ["no citation"]
    c = check_sentence("Opened items carry a 15% restocking fee.", ["S9"], SRC, 0.5)
    assert not c.supported and "S9" in c.issues[0]


def test_wrong_source_fails_on_wording():
    c = check_sentence("Opened items carry a restocking fee when returned.", ["S2"], SRC, 0.5)
    assert not c.supported and "key words" in c.issues[-1]


def _conf(question, sentences, answerable=True, idf=None):
    draft = Draft(answerable, sentences)
    checked = verify_draft(draft, list(SRC.values()))
    return score(question, checked, list(SRC.values()), answerable=answerable, idf=idf or {}, n_docs=40)


def test_confidence_high_when_grounded_and_on_topic():
    c = _conf("What is the restocking fee for opened items?",
              [{"text": "Opened items carry a 15% restocking fee.", "sources": ["S1"]}])
    assert c.level == "high" and c.value >= 0.9


def test_confidence_zero_when_model_says_not_answerable():
    assert _conf("anything", [], answerable=False).value == 0


def test_confidence_drops_when_sentences_are_removed():
    c = _conf("What is the restocking fee for opened items?", [
        {"text": "Opened items carry a 15% restocking fee.", "sources": ["S1"]},
        {"text": "Restocking is waived for VIP clients at 50%.", "sources": ["S1"]},
    ])
    assert c.grounded_share == 0.5 and c.value < 0.9
    assert any("failed the citation check" in r for r in c.reasons)


def test_off_topic_evidence_is_capped_low():
    c = _conf("When did Windows 10 reach end of support for laptops?",
              [{"text": "Deliveries to Sabah take 7 to 10 business days.", "sources": ["S2"]}])
    assert c.level == "low" and c.value <= c.coverage
