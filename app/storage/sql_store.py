from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.processing.arabic_utils import arabic_ratio
from app.processing.chunking import Chunk
from app.storage.models import Base, ChunkRecord, DocumentRecord

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


@contextmanager
def get_session():
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_document_with_chunks(
    session: Session,
    *,
    filename: str,
    file_type: str,
    page_count: int | None,
    strategy: str,
    contains_arabic: bool,
    chunks: list[Chunk],
) -> DocumentRecord:
    doc = DocumentRecord(
        filename=filename,
        file_type=file_type,
        page_count=page_count,
        chunking_strategy=strategy,
        contains_arabic=contains_arabic,
    )
    session.add(doc)
    session.flush()  # populate doc.id before attaching chunks

    for chunk in chunks:
        session.add(
            ChunkRecord(
                document_id=doc.id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                strategy=chunk.strategy,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                heading_path=" > ".join(chunk.heading_path) if chunk.heading_path else None,
                arabic_ratio=arabic_ratio(chunk.text),
            )
        )
    return doc


def list_documents(session: Session) -> list[DocumentRecord]:
    return session.query(DocumentRecord).order_by(DocumentRecord.ingested_at.desc()).all()


def get_chunks_by_ids(session: Session, chunk_ids: list[str]) -> list[ChunkRecord]:
    if not chunk_ids:
        return []
    return session.query(ChunkRecord).filter(ChunkRecord.id.in_(chunk_ids)).all()
