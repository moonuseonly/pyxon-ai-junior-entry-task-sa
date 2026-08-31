"""Decides which chunking strategy fits a document, instead of applying one
strategy to everything.

The signal is structural, not semantic (no LLM call needed just to pick a
strategy — that would be a lot of latency/cost for a decision that can be made
from cheap statistics):

- Heading density: many short heading blocks relative to paragraph blocks
  suggests a document with real internal structure (a book, a spec with
  sections) → dynamic chunking, so chunks respect those boundaries.
- Paragraph length variance: uniform paragraph lengths (a form, a templated
  report) don't benefit from structure-aware chunking and are cheaper to
  handle with fixed-size windows.
"""

from dataclasses import dataclass
from statistics import mean, pstdev

from app.parsers.base import ParsedDocument


@dataclass
class AnalysisResult:
    strategy: str  # "fixed" | "dynamic"
    heading_ratio: float
    length_variance_coefficient: float
    reason: str


def analyze(document: ParsedDocument) -> AnalysisResult:
    paragraph_blocks = [b for b in document.blocks if b.block_type == "paragraph"]
    heading_blocks = [b for b in document.blocks if b.block_type == "heading"]

    total_blocks = len(document.blocks) or 1
    heading_ratio = len(heading_blocks) / total_blocks

    if len(paragraph_blocks) >= 2:
        lengths = [len(b.text) for b in paragraph_blocks]
        avg_len = mean(lengths)
        variance_coefficient = (pstdev(lengths) / avg_len) if avg_len else 0.0
    else:
        variance_coefficient = 0.0

    # Either a meaningful density of headings, or highly irregular paragraph
    # lengths, indicates a document worth chunking structure-aware rather than
    # by a blind fixed window.
    if heading_ratio > 0.03 or variance_coefficient > 0.6:
        return AnalysisResult(
            strategy="dynamic",
            heading_ratio=heading_ratio,
            length_variance_coefficient=variance_coefficient,
            reason=(
                f"heading_ratio={heading_ratio:.2f} and/or "
                f"length_variance={variance_coefficient:.2f} indicate structural "
                "content — chunking will respect detected section boundaries."
            ),
        )

    return AnalysisResult(
        strategy="fixed",
        heading_ratio=heading_ratio,
        length_variance_coefficient=variance_coefficient,
        reason=(
            f"heading_ratio={heading_ratio:.2f} and length_variance="
            f"{variance_coefficient:.2f} both low — content reads as uniform, "
            "fixed-size windows are cheaper and sufficient."
        ),
    )
