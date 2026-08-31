"""PDF parsing with layout-aware heading detection.

PDFs carry no semantic markup — no "this is a heading" tag the way DOCX has
Word styles. What they do carry is font metadata per character. Rather than
guess headings from text patterns (short line, no trailing punctuation — a
guess that breaks constantly), this extracts each line's font size and
compares it against *that document's own* median body-text size, since font
sizes vary a lot between documents and a fixed pt threshold would misfire on
most PDFs it wasn't tuned for.

Known limitation: line grouping assumes a single-column layout (lines are
grouped purely by vertical position). Multi-column PDFs would need x-position
clustering too — noted here rather than silently mishandled.
"""

import statistics

import pdfplumber

from app.parsers.base import ParsedBlock, ParsedDocument

# A line whose median font size is this many times the document's body size
# (or more) is treated as a heading. Two tiers gives level-1 vs level-2
# headings without needing per-document calibration.
_LEVEL_1_RATIO = 1.3
_LEVEL_2_RATIO = 1.12
_HEADING_MAX_WORDS = 14


def _group_words_into_lines(words: list[dict]) -> list[list[dict]]:
    """Group words into lines by vertical position.

    2pt tolerance absorbs the sub-pixel 'top' jitter pdfplumber reports for
    words that are visually on the same line but not perfectly aligned.
    """
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        if lines and abs(word["top"] - lines[-1][0]["top"]) <= 2:
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines


def _line_text(line: list[dict]) -> str:
    return " ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"]))


def _line_font_size(line: list[dict]) -> float:
    return statistics.median(w["size"] for w in line)


def _classify_line(text: str, size: float, body_size: float) -> tuple[str, int]:
    ratio = (size / body_size) if body_size else 1.0
    short_enough = len(text.split()) <= _HEADING_MAX_WORDS
    if short_enough and ratio >= _LEVEL_1_RATIO:
        return "heading", 1
    if short_enough and ratio >= _LEVEL_2_RATIO:
        return "heading", 2
    return "paragraph", 0


class PDFParser:
    def parse(self, file_path: str) -> ParsedDocument:
        blocks: list[ParsedBlock] = []
        full_text_parts: list[str] = []

        with pdfplumber.open(file_path) as pdf:
            # Pass 1: gather every line's font size across the whole document
            # first, so heading detection is calibrated to *this* PDF's own
            # body-text size rather than a fixed point-size guess.
            lines_by_page: list[list[list[dict]]] = []
            all_body_sizes: list[float] = []

            for page in pdf.pages:
                words = page.extract_words(extra_attrs=["size"])
                lines = _group_words_into_lines(words)
                lines_by_page.append(lines)
                all_body_sizes.extend(_line_font_size(line) for line in lines if line)

            body_size = statistics.median(all_body_sizes) if all_body_sizes else 10.0

            # Pass 2: classify each line relative to that calibrated body size.
            for page_idx, lines in enumerate(lines_by_page, start=1):
                page_text_parts: list[str] = []
                for line in lines:
                    if not line:
                        continue
                    text = _line_text(line).strip()
                    if not text:
                        continue

                    size = _line_font_size(line)
                    block_type, level = _classify_line(text, size, body_size)
                    blocks.append(
                        ParsedBlock(
                            text=text,
                            block_type=block_type,
                            level=level,
                            page=page_idx,
                            metadata={"font_size": round(size, 1)},
                        )
                    )
                    page_text_parts.append(text)

                full_text_parts.append("\n".join(page_text_parts))

        return ParsedDocument(
            filename=file_path.split("/")[-1],
            file_type="pdf",
            blocks=blocks,
            full_text="\n".join(full_text_parts),
            page_count=len(lines_by_page),
        )
