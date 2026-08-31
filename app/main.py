import os

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.storage.sql_store import init_db

app = FastAPI(
    title=settings.app_name,
    description=(
        "Parses PDF/DOCX/TXT documents, picks a chunking strategy based on "
        "document structure, stores embeddings + metadata, and answers "
        "questions over ingested documents — with Arabic diacritics preserved "
        "by default."
    ),
    version="0.1.0",
)

app.include_router(router)


@app.on_event("startup")
def on_startup():
    os.makedirs(os.path.dirname(settings.chroma_persist_dir) or ".", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    init_db()
