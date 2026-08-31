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
