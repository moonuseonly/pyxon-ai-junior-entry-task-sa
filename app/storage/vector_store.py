from functools import lru_cache

import chromadb
from fastembed import TextEmbedding

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    # Multilingual ONNX embedding model for Arabic and English.
    return TextEmbedding(
        model_name=settings.embedding_model
    )


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

    MiniLM does not require E5-style query:/passage: prefixes.
    """

    if not texts:
        return []

    embedder = get_embedder()

    return [
        embedding.tolist()
        for embedding in embedder.embed(texts)
    ]


def add_chunks(
    *,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:

    if not ids:
        return

    collection = get_collection()

    # Process chunks in small batches to reduce RAM usage.
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
