"""PDF parsing with layout-aware heading detection.

PDFs carry no semantic markup — no "this is a heading" tag the way DOCX has
Word styles. What they do carry is font metadata per character. Rather than
guess headings from text patterns, this extracts each line's font size and
compares it against the document's own median body-text size.

The parser uses two passes:

1. Pass 1 collects only font-size values needed for calibration.
2. Pass 2 processes one page at a time, so all word dictionaries are not
   kept in memory simultaneously.

Known limitation: line grouping assumes a single-column layout.
"""

import statistics
from pathlib import Path

import pdfplumber

from app.parsers.base import ParsedBlock, ParsedDocument
from app.processing.arabic_utils import contains_arabic


_LEVEL_1_RATIO = 1.3
_LEVEL_2_RATIO = 1.12
_HEADING_MAX_WORDS = 14


def _group_words_into_lines(words: list[dict]) -> list[list[dict]]:
    """Group words into lines by vertical position."""

    lines: list[list[dict]] = []

    for word in sorted(
        words,
        key=lambda w: (round(w["top"], 1), w["x0"]),
    ):
        if (
            lines
            and abs(word["top"] - lines[-1][0]["top"]) <= 2
        ):
            lines[-1].append(word)
        else:
            lines.append([word])

    return lines


def _line_text(line: list[dict]) -> str:
    """Reconstruct a line in visual reading order."""

    joined_for_detection = "".join(
        w["text"] for w in line
    )

    reverse = contains_arabic(joined_for_detection)

    ordered = sorted(
        line,
        key=lambda w: w["x0"],
        reverse=reverse,
    )

    return " ".join(
        w["text"] for w in ordered
    )


def _line_font_size(line: list[dict]) -> float:
    """Return the median font size of a line."""

    return statistics.median(
        w["size"] for w in line
    )


def _classify_line(
    text: str,
    size: float,
    body_size: float,
) -> tuple[str, int]:

    ratio = (
        size / body_size
        if body_size
        else 1.0
    )

    short_enough = (
        len(text.split())
        <= _HEADING_MAX_WORDS
    )

    if (
        short_enough
        and ratio >= _LEVEL_1_RATIO
    ):
        return "heading", 1

    if (
        short_enough
        and ratio >= _LEVEL_2_RATIO
    ):
        return "heading", 2

    return "paragraph", 0


class PDFParser:

    def parse(
        self,
        file_path: str,
    ) -> ParsedDocument:

        blocks: list[ParsedBlock] = []
        full_text_parts: list[str] = []

        # ---------------------------------------------------------
        # PASS 1
        # Collect only font-size values.
        #
        # We deliberately do NOT store all PDF words/lines here.
        # ---------------------------------------------------------

        all_body_sizes: list[float] = []

        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:

                words = page.extract_words(
                    extra_attrs=["size"]
                )

                lines = _group_words_into_lines(
                    words
                )

                all_body_sizes.extend(
                    _line_font_size(line)
                    for line in lines
                    if line
                )

                # Release page-specific objects before
                # moving to the next page.
                del lines
                del words

            body_size = (
                statistics.median(all_body_sizes)
                if all_body_sizes
                else 10.0
            )

        # Font-size values are no longer needed after
        # calibration.
        del all_body_sizes

        # ---------------------------------------------------------
        # PASS 2
        # Process ONE page at a time.
        # ---------------------------------------------------------

        with pdfplumber.open(file_path) as pdf:

            for page_idx, page in enumerate(
                pdf.pages,
                start=1,
            ):

                words = page.extract_words(
                    extra_attrs=["size"]
                )

                lines = _group_words_into_lines(
                    words
                )

                page_text_parts: list[str] = []

                for line in lines:

                    if not line:
                        continue

                    text = _line_text(line).strip()

                    if not text:
                        continue

                    size = _line_font_size(line)

                    block_type, level = _classify_line(
                        text,
                        size,
                        body_size,
                    )

                    blocks.append(
                        ParsedBlock(
                            text=text,
                            block_type=block_type,
                            level=level,
                            page=page_idx,
                            metadata={
                                "font_size": round(
                                    size,
                                    1,
                                )
                            },
                        )
                    )

                    page_text_parts.append(text)

                full_text_parts.append(
                    "\n".join(page_text_parts)
                )

                # Explicitly release page data.
                del lines
                del words
                del page_text_parts

        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="pdf",
            blocks=blocks,
            full_text="\n".join(full_text_parts),
            page_count=len(full_text_parts),
        )
