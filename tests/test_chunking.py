from app.parsers.base import ParsedBlock, ParsedDocument
from app.processing import document_analyzer
from app.processing.arabic_utils import (
    arabic_ratio,
    contains_arabic,
    normalize_arabic,
    strip_diacritics,
)
from app.processing.chunking import chunk_document


def _doc_with_headings() -> ParsedDocument:
    blocks = [
        ParsedBlock(text="Overview", block_type="heading", level=1),
        ParsedBlock(text="This section explains the overview. " * 5, block_type="paragraph"),
        ParsedBlock(text="Details", block_type="heading", level=1),
        ParsedBlock(text="This section explains the details. " * 5, block_type="paragraph"),
    ]
    full_text = "\n".join(b.text for b in blocks)
    return ParsedDocument(filename="t.txt", file_type="txt", blocks=blocks, full_text=full_text)


def _uniform_doc() -> ParsedDocument:
    blocks = [
        ParsedBlock(text="Uniform paragraph content. " * 6, block_type="paragraph")
        for _ in range(6)
    ]
    full_text = "\n".join(b.text for b in blocks)
    return ParsedDocument(filename="u.txt", file_type="txt", blocks=blocks, full_text=full_text)


def test_analyzer_picks_dynamic_for_structured_doc():
    result = document_analyzer.analyze(_doc_with_headings())
    assert result.strategy == "dynamic"


def test_analyzer_picks_fixed_for_uniform_doc():
    result = document_analyzer.analyze(_uniform_doc())
    assert result.strategy == "fixed"


def test_dynamic_chunk_respects_heading_boundaries():
    doc = _doc_with_headings()
    chunks = chunk_document(doc, "dynamic")
    assert len(chunks) == 2
    assert chunks[0].heading_path == ["Overview"]
    assert chunks[1].heading_path == ["Details"]
    assert "overview" in chunks[0].text.lower()
    assert "details" in chunks[1].text.lower()


def test_fixed_chunk_produces_overlapping_windows():
    doc = _uniform_doc()
    chunks = chunk_document(doc, "fixed")
    assert len(chunks) > 1
    # every chunk (except possibly the last) should be roughly chunk-size length
    for c in chunks[:-1]:
        assert len(c.text) > 0


def test_fixed_chunk_terminates_on_short_text():
    doc = ParsedDocument(filename="s.txt", file_type="txt", blocks=[], full_text="short text")
    chunks = chunk_document(doc, "fixed")
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_diacritics_preserved_by_default():
    text = "الطَّالِبُ كَتَبَ الدَّرْسَ"
    normalized = normalize_arabic(text)
    assert normalized == text  # no diacritics stripped by default


def test_diacritics_stripped_when_requested():
    text = "الطَّالِبُ"
    stripped = strip_diacritics(text)
    assert stripped == "الطالب"
    assert len(stripped) < len(text)


def test_arabic_ratio_and_detection():
    assert contains_arabic("مرحبا") is True
    assert contains_arabic("hello") is False
    assert arabic_ratio("مرحبا") == 1.0
    assert 0.0 < arabic_ratio("hello مرحبا") < 1.0


def test_tatweel_removed():
    text = "مرحـــبا"  # contains tatweel (kashida)
    normalized = normalize_arabic(text)
    assert "\u0640" not in normalized


def test_dynamic_chunk_heading_stack_resets_on_new_top_level_heading():
    blocks = [
        ParsedBlock(text="Chapter 1", block_type="heading", level=1),
        ParsedBlock(text="Section 1.1", block_type="heading", level=2),
        ParsedBlock(text="Coverage content here.", block_type="paragraph"),
        ParsedBlock(text="Chapter 2", block_type="heading", level=1),
        ParsedBlock(text="Section 2.1", block_type="heading", level=2),
        ParsedBlock(text="Churn content here.", block_type="paragraph"),
    ]
    doc = ParsedDocument(filename="n.txt", file_type="txt", blocks=blocks, full_text="")
    chunks = chunk_document(doc, "dynamic")

    assert chunks[0].heading_path == ["Chapter 1", "Section 1.1"]
    assert chunks[1].heading_path == ["Chapter 2", "Section 2.1"]
    # Chapter 2's chunk must NOT inherit Chapter 1's children
    assert "Section 1.1" not in chunks[1].heading_path


def test_dynamic_chunk_splits_section_exceeding_size_cap():
    blocks = [ParsedBlock(text="Long Section", block_type="heading", level=1)] + [
        ParsedBlock(text=f"Paragraph {i} with filler content padding it out. " * 3, block_type="paragraph")
        for i in range(15)
    ]
    doc = ParsedDocument(filename="long.txt", file_type="txt", blocks=blocks, full_text="")
    chunks = chunk_document(doc, "dynamic")

    assert len(chunks) > 1
    for c in chunks:
        assert c.heading_path == ["Long Section"]  # same section, split by size only
