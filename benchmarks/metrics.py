"""Retrieval and chunking quality metrics for the benchmark suite."""

import numpy as np


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def chunk_coherence_score(chunk_embeddings: list[np.ndarray]) -> float:
    """Proxy for chunk coherence: average pairwise cosine similarity of
    sentence embeddings *within* a chunk. Higher means the chunk's sentences
    are semantically closer together, i.e. the chunk boundary didn't stitch
    together unrelated content.
    """
    if len(chunk_embeddings) < 2:
        return 1.0
    vectors = np.array(chunk_embeddings)
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sim_matrix = norm @ norm.T
    n = len(vectors)
    upper_triangle_sum = sim_matrix[np.triu_indices(n, k=1)].sum()
    pair_count = n * (n - 1) / 2
    return float(upper_triangle_sum / pair_count)


def boundary_contrast(
    within_chunk_similarity: float, cross_boundary_similarity: float
) -> float:
    """How much more coherent chunks are internally vs. across their
    boundaries. Positive and larger is better — it means the chunker is
    cutting at genuine topic shifts rather than arbitrarily.
    """
    return within_chunk_similarity - cross_boundary_similarity
