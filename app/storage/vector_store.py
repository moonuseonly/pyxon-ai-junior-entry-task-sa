from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    # Multilingual model so Arabic and English chunks share one embedding
    # space — a query in English can retrieve an Arabic chunk and vice versa,
    # which a monolingual English model can't do.
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    return embedder.encode(texts, normalize_embeddings=True).tolist()


def add_chunks(
    *,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:
    if not ids:
        return
    collection = get_collection()
    embeddings = embed_texts(texts)
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def query(text: str, top_k: int) -> dict:
    collection = get_collection()
    embedding = embed_texts([text])[0]
    return collection.query(query_embeddings=[embedding], n_results=top_k)
