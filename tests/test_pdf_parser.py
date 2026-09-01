import pytest
from fpdf import FPDF

from app.parsers.pdf_parser import PDFParser


@pytest.fixture
def sample_pdf_path(tmp_path):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Quarterly Churn Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "Customer churn increased by twelve percent in the last quarter.")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Methodology", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "We used a random forest classifier trained on usage data.")

    path = tmp_path / "sample.pdf"
    pdf.output(str(path))
    return str(path)


def test_pdf_parser_detects_title_as_level_1_heading(sample_pdf_path):
    doc = PDFParser().parse(sample_pdf_path)
    headings = [b for b in doc.blocks if b.block_type == "heading"]
    assert any(h.text == "Quarterly Churn Report" and h.level == 1 for h in headings)


def test_pdf_parser_detects_section_heading_as_level_2(sample_pdf_path):
    doc = PDFParser().parse(sample_pdf_path)
    headings = [b for b in doc.blocks if b.block_type == "heading"]
    assert any(h.text == "Methodology" and h.level == 2 for h in headings)


def test_pdf_parser_body_text_not_classified_as_heading(sample_pdf_path):
    doc = PDFParser().parse(sample_pdf_path)
    paragraphs = [b for b in doc.blocks if b.block_type == "paragraph"]
    assert any("random forest classifier" in p.text for p in paragraphs)


def test_pdf_line_text_reads_right_to_left_for_arabic():
    # PDF word coordinates are visual (on-page) position, not reading order.
    # For an RTL line, the rightmost word on the page is read FIRST.
    from app.parsers.pdf_parser import _line_text

    arabic_words = [
        {"text": "الدرس", "x0": 10},    # leftmost on page -> read last
        {"text": "كتب", "x0": 60},       # middle
        {"text": "الطالب", "x0": 110},   # rightmost on page -> read first
    ]
    assert _line_text(arabic_words) == "الطالب كتب الدرس"


def test_pdf_line_text_reads_left_to_right_for_english():
    from app.parsers.pdf_parser import _line_text

    english_words = [
        {"text": "lesson", "x0": 110},
        {"text": "the", "x0": 60},
        {"text": "wrote", "x0": 10},
    ]
    assert _line_text(english_words) == "wrote the lesson"


def test_pdf_parser_extracts_filename_not_full_path(tmp_path):
    # Regression test: filename extraction used to do file_path.split("/")[-1],
    # which returns the ENTIRE path unchanged on Windows (backslash-separated
    # paths have no "/" to split on). Nested tmp_path already exercises this
    # on whatever platform the test runs on.
    nested_dir = tmp_path / "some" / "nested" / "folder"
    nested_dir.mkdir(parents=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 10, "minimal content")
    path = nested_dir / "my_report.pdf"
    pdf.output(str(path))

    doc = PDFParser().parse(str(path))
    assert doc.filename == "my_report.pdf"
    assert "\\" not in doc.filename and "/" not in doc.filename


def test_pdf_parser_records_page_count(sample_pdf_path):
    doc = PDFParser().parse(sample_pdf_path)
    assert doc.page_count == 1


def test_pdf_parser_heading_ratio_relative_to_body_size_not_fixed(sample_pdf_path):
    # The classification should be driven by the ratio to *this document's*
    # body size, not a hardcoded point value — this test just documents that
    # contract by checking font_size metadata was actually captured per line.
    doc = PDFParser().parse(sample_pdf_path)
    sizes = {b.metadata.get("font_size") for b in doc.blocks}
    assert 18.0 in sizes
    assert 14.0 in sizes
    assert 11.0 in sizes
    