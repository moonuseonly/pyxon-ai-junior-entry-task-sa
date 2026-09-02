from functools import lru_cache

import chromadb
from fastembed import TextEmbedding

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    # Multilingual, ONNX-based (no PyTorch) so Arabic and English chunks
    # share one embedding space without the memory cost of a torch model.
    return TextEmbedding(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    # E5 models are trained with "query:"/"passage:" prefixes — using them
    # correctly noticeably improves retrieval quality over leaving them off.
    prefix = "query" if is_query else "passage"
    prefixed = [f"{prefix}: {t}" for t in texts]
    embedder = get_embedder()
    return [e.tolist() for e in embedder.embed(prefixed)]


def add_chunks(
    *,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:
    if not ids:
        return
    collection = get_collection()
    embeddings = embed_texts(texts, is_query=False)
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def query(text: str, top_k: int) -> dict:
    collection = get_collection()
    embedding = embed_texts([text], is_query=True)[0]
    return collection.query(query_embeddings=[embedding], n_results=top_k)
