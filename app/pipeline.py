"""Wires parsing -> analysis -> chunking -> dual storage into one call.

This is the module the API (and the benchmark suite, and the CLI demo script)
all call into, so ingestion behaves identically no matter what triggers it.
"""

from dataclasses import dataclass

from app.parsers.registry import parse_document
from app.processing import document_analyzer
from app.processing.arabic_utils import arabic_ratio
from app.processing.chunking import chunk_document
from app.storage import vector_store
from app.storage.models import DocumentRecord
from app.storage.sql_store import get_session, save_document_with_chunks


@dataclass
class IngestResult:
    document: DocumentRecord
    chunk_count: int
    strategy: str
    reason: str


def ingest_file(
    file_path: str,
    *,
    display_filename: str | None = None,
) -> IngestResult:

    parsed = parse_document(file_path)

    if display_filename:
        parsed.filename = display_filename

    analysis = document_analyzer.analyze(parsed)

    chunks = chunk_document(
        parsed,
        analysis.strategy,
    )

    with get_session() as session:

        doc = save_document_with_chunks(
            session,
            filename=parsed.filename,
            file_type=parsed.file_type,
            page_count=parsed.page_count,
            strategy=analysis.strategy,
            contains_arabic=arabic_ratio(parsed.full_text) > 0.05,
            chunks=chunks,
        )

        session.flush()

        chunk_records = sorted(
            doc.chunks,
            key=lambda c: c.chunk_index,
        )

        # ---------------------------------------------------------
        # Add vectors in small batches.
        #
        # We deliberately avoid building the complete ids/texts/
        # metadatas payload at once.
        # ---------------------------------------------------------

        batch_size = 16

        for start in range(
            0,
            len(chunk_records),
            batch_size,
        ):

            batch = chunk_records[
                start:start + batch_size
            ]

            vector_store.add_chunks(
                ids=[c.id for c in batch],
                texts=[c.text for c in batch],
                metadatas=[
                    {
                        "document_id": doc.id,
                        "filename": doc.filename,
                        "chunk_index": c.chunk_index,
                        "page_start": c.page_start or -1,
                        "page_end": c.page_end or -1,
                        "heading_path": c.heading_path or "",
                    }
                    for c in batch
                ],
            )

        chunk_count = len(chunk_records)

    return IngestResult(
        document=doc,
        chunk_count=chunk_count,
        strategy=analysis.strategy,
        reason=analysis.reason,
    )
