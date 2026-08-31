"""Hybrid retrieval: vector similarity + a lightweight keyword overlap signal.

Pure vector search misses exact-term matches that matter for names, codes, or
rare terms the embedding model wasn't trained to weight heavily. Rather than
pull in a full BM25 dependency for this, we combine cosine similarity from
Chroma with a cheap token-overlap score computed over the candidate set
returned by the vector search — good enough to noticeably help precision on
short, keyword-heavy queries without adding another moving part.
"""

from dataclasses import dataclass

from app.storage import vector_store


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    vector_score: float
    keyword_score: float
    combined_score: float


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in text.split() if len(tok) > 1}


def _keyword_overlap(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    overlap = query_tokens & text_tokens
    return len(overlap) / len(query_tokens)


def retrieve(question: str, top_k: int, *, keyword_weight: float = 0.25) -> list[RetrievedChunk]:
    raw = vector_store.query(question, top_k=max(top_k * 3, top_k))  # over-fetch, then rerank
    query_tokens = _tokenize(question)

    ids = raw.get("ids", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    candidates: list[RetrievedChunk] = []
    for cid, doc_text, meta, distance in zip(ids, documents, metadatas, distances):
        vector_score = 1.0 - distance  # cosine distance -> similarity
        keyword_score = _keyword_overlap(query_tokens, doc_text)
        combined = (1 - keyword_weight) * vector_score + keyword_weight * keyword_score
        candidates.append(
            RetrievedChunk(
                chunk_id=cid,
                text=doc_text,
                metadata=meta or {},
                vector_score=vector_score,
                keyword_score=keyword_score,
                combined_score=combined,
            )
        )

    candidates.sort(key=lambda c: c.combined_score, reverse=True)
    return candidates[:top_k]
