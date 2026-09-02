import os
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.parsers.registry import UnsupportedFileType
from app.pipeline import ingest_file
from app.rag.generation import generate_answer
from app.rag.retriever import retrieve
from app.storage.sql_store import get_session, list_documents

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}


@router.post("/ingest")
def ingest(file: UploadFile = File(...)):
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix.lstrip(".") not in {"pdf", "docx", "txt"}:
        raise HTTPException(400, "Unsupported file type. Use PDF, DOCX, or TXT.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_file(tmp_path, display_filename=file.filename)
    except UnsupportedFileType as e:
        raise HTTPException(400, str(e)) from e
    finally:
        os.unlink(tmp_path)

    return {
        "document_id": result.document.id,
        "filename": result.document.filename,
        "chunk_count": result.chunk_count,
        "strategy": result.strategy,
        "strategy_reason": result.reason,
    }


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=settings.default_top_k, ge=1, le=20)


@router.post("/query")
def query(req: QueryRequest):
    chunks = retrieve(req.question, top_k=req.top_k)
    result = generate_answer(req.question, chunks)
    return {
        "question": req.question,
        "answer": result["answer"],
        "mode": result["mode"],
        "note": result.get("note"),
        "sources": [
            {
                "chunk_id": c.chunk_id,
                "filename": c.metadata.get("filename"),
                "page_start": c.metadata.get("page_start"),
                "page_end": c.metadata.get("page_end"),
                "heading_path": c.metadata.get("heading_path"),
                "combined_score": round(c.combined_score, 4),
            }
            for c in chunks
        ],
    }


@router.get("/documents")
def documents():
    with get_session() as session:
        docs = list_documents(session)
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "file_type": d.file_type,
                "chunking_strategy": d.chunking_strategy,
                "contains_arabic": d.contains_arabic,
                "chunk_count": len(d.chunks),
                "ingested_at": d.ingested_at.isoformat(),
            }
            for d in docs
        ]
