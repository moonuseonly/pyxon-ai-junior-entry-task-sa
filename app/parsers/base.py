from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ParsedBlock:
    """One structural unit extracted from a source document.

    Parsers don't chunk — they extract structure. A PDF page, a DOCX paragraph,
    or a heading each becomes a block, tagged with what it is. The chunker
    downstream decides how to group blocks into chunks.
    """

    text: str
    block_type: str  # "heading" | "paragraph" | "table" | "list_item"
    level: int = 0   # heading level (1 = top), 0 for non-headings
    page: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    filename: str
    file_type: str  # "pdf" | "docx" | "txt"
    blocks: list[ParsedBlock]
    full_text: str
    page_count: int | None = None


class DocumentParser(Protocol):
    def parse(self, file_path: str) -> ParsedDocument: ...
