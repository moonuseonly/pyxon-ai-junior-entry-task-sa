from pathlib import Path

from app.parsers.base import ParsedBlock, ParsedDocument


class TXTParser:
    def parse(self, file_path: str) -> ParsedDocument:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            raw = f.read()

        blocks: list[ParsedBlock] = []
        # Plain text has no structural markup, so paragraphs are inferred from
        # blank-line separation — the one structural signal available.
        for para in raw.split("\n\n"):
            para = para.strip()
            if para:
                blocks.append(ParsedBlock(text=para, block_type="paragraph"))

        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="txt",
            blocks=blocks,
            full_text=raw,
        )
    
