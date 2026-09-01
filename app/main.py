import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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

_STATIC_DIR = Path(__file__).parent.parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_demo_ui():
    # Simple hand-built page instead of directing everyone to /docs — Swagger
    # is great for exploring the API but is a lot of technical surface area
    # for someone just trying to demo the project.
    return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.on_event("startup")
def on_startup():
    os.makedirs(os.path.dirname(settings.chroma_persist_dir) or ".", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    init_db()
    