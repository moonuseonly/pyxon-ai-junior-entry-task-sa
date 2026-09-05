import os
from functools import lru_cache

import chromadb
import cohere
from fastembed import TextEmbedding

from app.config import settings

# Optional hosted-embeddings path: if COHERE_API_KEY is set, embeddings are
# generated via Cohere's API instead of the local ONNX model. Mirrors the
# "env var present = optional service enabled" pattern GROQ_API_KEY uses in
# app/rag/generation.py. Local fastembed stays the real, self-contained
# implementation; Cohere only kicks in for the memory-constrained hosted
# demo, where the local model's RAM footprint doesn't fit.
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_EMBED_MODEL = "embed-multilingual-v3.0"


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    # Multilingual ONNX embedding model for Arabic and English.
    return TextEmbedding(
        model_name=settings.embedding_model,
        threads=1,
    )


@lru_cache(maxsize=1)
def get_cohere_client() -> cohere.ClientV2:
    return cohere.ClientV2(api_key=COHERE_API_KEY)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir
    )

    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(
    texts: list[str],
    *,
    is_query: bool = False,
) -> list[list[float]]:
    """
    Generate embeddings for Arabic/English text.

    Uses Cohere's hosted multilingual embeddings when COHERE_API_KEY is set
    (the hosted demo's path around Render's free-tier RAM limit). Otherwise
    runs the local ONNX model — MiniLM does not require E5-style
    query:/passage: prefixes, so is_query only matters for the Cohere path.
    """

    if not texts:
        return []

    if COHERE_API_KEY:
        client = get_cohere_client()
        response = client.embed(
            texts=texts,
            model=COHERE_EMBED_MODEL,
            input_type="search_query" if is_query else "search_document",
            embedding_types=["float"],
        )
        return [list(vec) for vec in response.embeddings.float_]

    embedder = get_embedder()

    return [
        embedding.tolist()
        for embedding in embedder.embed(texts)
    ]


def clear_all() -> None:
    """Wipe all stored vectors. Keeps the demo to a single active document
    and stops unbounded growth across test uploads."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        client.delete_collection(name=settings.chroma_collection)
    except chromadb.errors.NotFoundError:
        pass
    get_collection.cache_clear()


def add_chunks(
    *,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:

    if not ids:
        return

    collection = get_collection()

    batch_size = 16

    for start in range(0, len(ids), batch_size):

        end = start + batch_size

        batch_ids = ids[start:end]
        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]

        embeddings = embed_texts(
            batch_texts,
            is_query=False,
        )

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas,
        )


def query(text: str, top_k: int) -> dict:

    collection = get_collection()

    embedding = embed_texts(
        [text],
        is_query=True,
    )[0]

    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
    )
