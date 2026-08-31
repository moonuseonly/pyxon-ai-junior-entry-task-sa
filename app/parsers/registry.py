from app.parsers.base import DocumentParser, ParsedDocument
from app.parsers.docx_parser import DOCXParser
from app.parsers.pdf_parser import PDFParser
from app.parsers.txt_parser import TXTParser

_PARSERS: dict[str, DocumentParser] = {
    "pdf": PDFParser(),
    "docx": DOCXParser(),
    "txt": TXTParser(),
}


class UnsupportedFileType(ValueError):
    pass


def parse_document(file_path: str) -> ParsedDocument:
    ext = file_path.rsplit(".", 1)[-1].lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise UnsupportedFileType(
            f"'{ext}' is not supported. Supported types: {', '.join(_PARSERS)}"
        )
    return parser.parse(file_path)
