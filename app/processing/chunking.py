from dataclasses import dataclass, field

from app.config import settings
from app.parsers.base import ParsedBlock, ParsedDocument
from app.processing.arabic_utils import normalize_arabic, safe_split_point


@dataclass
class Chunk:
    text: str
    chunk_index: int
    strategy: str  # "fixed" | "dynamic"
    page_start: int | None = None
    page_end: int | None = None
    heading_path: list[str] = field(default_factory=list)


def _normalize_block_text(text: str) -> str:
    return normalize_arabic(text, strip_diacritics_=settings.strip_diacritics_default)


def fixed_chunk(document: ParsedDocument) -> list[Chunk]:
    """Fixed-size sliding-window chunking with overlap.

    Word/diacritic-safe: boundaries snap to whitespace via `safe_split_point`
    rather than cutting mid-token, which matters more for Arabic (splitting a
    base letter from its diacritic changes what re-reads correctly) than for
    English but is good hygiene either way.
    """
    text = _normalize_block_text(document.full_text)
    size = settings.fixed_chunk_size
    overlap = settings.fixed_chunk_overlap

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        target_end = min(start + size, len(text))
        end = safe_split_point(text, target_end)
        piece = text[start:end].strip()
        if len(piece) >= settings.min_chunk_size or end == len(text):
            if piece:
                chunks.append(Chunk(text=piece, chunk_index=idx, strategy="fixed"))
                idx += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # guarantee forward progress
    return chunks


def _flush_group(
    group_blocks: list[ParsedBlock], heading_path: list[str], idx: int
) -> Chunk | None:
    if not group_blocks:
        return None
    text = _normalize_block_text("\n".join(b.text for b in group_blocks))
    if not text.strip():
        return None
    pages = [b.page for b in group_blocks if b.page is not None]
    return Chunk(
        text=text,
        chunk_index=idx,
        strategy="dynamic",
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        heading_path=list(heading_path),
    )


def dynamic_chunk(document: ParsedDocument) -> list[Chunk]:
    """Structure-aware chunking that groups blocks under their nearest heading
    and only splits a section further if it exceeds `dynamic_chunk_max_size`.

    Unlike fixed chunking, this never cuts a heading's content in half unless
    that section is itself larger than the soft cap — in which case it falls
    back to safe-split-point sub-chunking within the section.
    """
    chunks: list[Chunk] = []
    heading_stack: list[str] = []
    current_group: list[ParsedBlock] = []
    idx = 0

    def current_group_len() -> int:
        return sum(len(b.text) for b in current_group)

    for block in document.blocks:
        if block.block_type == "heading":
            flushed = _flush_group(current_group, heading_stack, idx)
            if flushed:
                chunks.append(flushed)
                idx += 1
            current_group = []

            level = max(block.level, 1)
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(block.text)
            continue

        current_group.append(block)
        if current_group_len() >= settings.dynamic_chunk_max_size:
            flushed = _flush_group(current_group, heading_stack, idx)
            if flushed:
                chunks.append(flushed)
                idx += 1
            current_group = []

    flushed = _flush_group(current_group, heading_stack, idx)
    if flushed:
        chunks.append(flushed)

    # A document with no detected headings at all degenerates to one giant
    # group — fall back to fixed chunking rather than emitting a single
    # oversized chunk that would be useless for retrieval.
    if len(chunks) <= 1 and len(document.full_text) > settings.dynamic_chunk_max_size:
        return fixed_chunk(document)

    return chunks


def chunk_document(document: ParsedDocument, strategy: str) -> list[Chunk]:
    if strategy == "dynamic":
        return dynamic_chunk(document)
    return fixed_chunk(document)
