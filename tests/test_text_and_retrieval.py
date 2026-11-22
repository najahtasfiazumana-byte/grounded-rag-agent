from grounded_rag.documents import chunk_markdown
from grounded_rag.text import expand_terms, load_glossary, numbers_in, split_sentences, stem, tokenize


def test_stemmer_matches_word_forms():
    assert stem("approves") == stem("approval") == stem("approved") == stem("approve")
    assert stem("returns") == "return" and stem("days") == "day"
    assert stem("sent") == stem("send")
    assert stem("business") == "business"


def test_numbers_are_percent_aware_and_ignore_separators():
    assert numbers_in("a 15% fee on MYR 3,000") == {"15%", "3000"}
    assert "15" not in numbers_in("15% off")
    assert numbers_in("costs 1.5% per month") == {"1.5%"}


def test_tokenize_drops_question_words_and_citations():
    assert tokenize("How long is the warranty? [S1]") == ["warranty"]


def test_glossary_expands_abbreviations():
    glossary = load_glossary("data/glossary.yaml")
    alts = expand_terms({"m365"}, glossary)["m365"]
    assert {"microsoft", "365"} <= alts


def test_sentence_split():
    assert split_sentences("One. Two 3.5% rate! Three?") == ["One.", "Two 3.5% rate!", "Three?"]


def test_chunking_by_heading_with_stable_ids_and_no_disclaimers():
    md = "# Title\n\n> disclaimer\n\n## First part\n\nAlpha.\n\n## Second part\n\nBeta."
    chunks = chunk_markdown(md, "doc", "kb/doc.md")
    assert [c.id for c in chunks] == ["doc#first-part", "doc#second-part"]
    assert chunks[0].doc_title == "Title" and "disclaimer" not in chunks[0].text
    assert chunks[1].label == "Title › Second part"


def test_long_sections_split_with_overlap():
    para = " ".join(["word"] * 100)
    md = f"## Big\n\n{para}\n\n{para}\n\n{para}"
    chunks = chunk_markdown(md, "doc", "doc.md")
    assert len(chunks) >= 2 and chunks[1].id == "doc#big-2"


def test_index_finds_the_right_section(index):
    cases = {
        "What is the restocking fee for opened items?": "warranty_and_returns#returns-and-restocking-fee",
        "How long does delivery to Sabah take?": "delivery_and_sla#delivery-times",
        "What is the late payment charge?": "payment_terms#late-payment",
        "Which laptop for a graphic designer?": "product_catalog_notes#recommended-business-laptops",
    }
    for q, expected in cases.items():
        assert index.search(q, k=3)[0].chunk.id == expected, q


def test_glossary_helps_retrieval(index):
    glossary = load_glossary("data/glossary.yaml")
    ids = [h.chunk.id for h in index.search("M365 renewal quote timeline", k=3, glossary=glossary)]
    assert "m365_renewals_playbook#renewal-timeline" in ids


def test_unrelated_question_finds_nothing(index):
    assert index.search("Who won the World Cup?", k=3) == []


def test_index_save_and_load_roundtrip(index, tmp_path):
    from grounded_rag.index import Index

    path = tmp_path / "index.json"
    index.save(path)
    loaded = Index.load(path)
    assert len(loaded.chunks) == len(index.chunks)
    assert loaded.search("restocking fee", 1)[0].chunk.id == index.search("restocking fee", 1)[0].chunk.id
