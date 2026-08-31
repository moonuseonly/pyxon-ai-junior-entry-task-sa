"""Benchmark runner.

Usage:
    python -m benchmarks.run_benchmark

Ingests the eval-set documents into an isolated store (separate from your
real data), runs each query, and reports retrieval quality, chunk coherence,
latency, and an Arabic diacritics comparison.
"""

import json
import os
import shutil
import time
from pathlib import Path

# Point the app at an isolated benchmark store BEFORE importing app modules,
# since settings are read once at import time.
_BENCH_DIR = Path(__file__).parent / "data" / "_bench_store"
os.environ["APP_DATABASE_URL"] = f"sqlite:///{_BENCH_DIR / 'bench.db'}"
os.environ["APP_CHROMA_PERSIST_DIR"] = str(_BENCH_DIR / "chroma")
os.environ["APP_CHROMA_COLLECTION"] = "benchmark"

from app.pipeline import ingest_file  # noqa: E402
from app.processing.arabic_utils import strip_diacritics  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402
from app.storage.sql_store import init_db  # noqa: E402
from app.storage.vector_store import get_embedder  # noqa: E402
from benchmarks.metrics import (  # noqa: E402
    chunk_coherence_score,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

TOP_K = 3


def _reset_store():
    shutil.rmtree(_BENCH_DIR, ignore_errors=True)
    _BENCH_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


def _write_temp_doc(tmp_dir: Path, filename: str, text: str) -> str:
    path = tmp_dir / filename
    path.write_text(text, encoding="utf-8")
    return str(path)


def run():
    eval_set = json.loads((Path(__file__).parent / "data" / "eval_set.json").read_text())
    _reset_store()

    tmp_dir = _BENCH_DIR / "source_docs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Ingestion ---
    filename_to_doc_id = {}
    ingest_latencies = []
    for doc in eval_set["documents"]:
        path = _write_temp_doc(tmp_dir, doc["filename"], doc["text"])
        start = time.perf_counter()
        result = ingest_file(path)
        ingest_latencies.append(time.perf_counter() - start)
        filename_to_doc_id[doc["filename"]] = result.document.id
        print(
            f"  ingested {doc['filename']:35s} strategy={result.strategy:8s} "
            f"chunks={result.chunk_count}"
        )

    # --- Retrieval quality ---
    print("\n--- Retrieval quality (top_k={}) ---".format(TOP_K))
    precisions, recalls, mrrs = [], [], []
    query_latencies = []
    for q in eval_set["queries"]:
        start = time.perf_counter()
        results = retrieve(q["question"], top_k=TOP_K)
        query_latencies.append(time.perf_counter() - start)

        retrieved_filenames = [r.metadata.get("filename") for r in results]
        expected_doc_id = filename_to_doc_id[q["expected_filename"]]
        relevant_ids = {expected_doc_id}
        retrieved_doc_ids = [r.metadata.get("document_id") for r in results]

        p = precision_at_k(retrieved_doc_ids, relevant_ids, TOP_K)
        r = recall_at_k(retrieved_doc_ids, relevant_ids, TOP_K)
        mrr = mean_reciprocal_rank(retrieved_doc_ids, relevant_ids)
        precisions.append(p)
        recalls.append(r)
        mrrs.append(mrr)

        hit = "✓" if q["expected_filename"] in retrieved_filenames else "✗"
        print(f"  [{hit}] {q['question']!r} -> top match: {retrieved_filenames[:1]}")

    print(f"\n  Precision@{TOP_K}: {sum(precisions)/len(precisions):.2f}")
    print(f"  Recall@{TOP_K}:    {sum(recalls)/len(recalls):.2f}")
    print(f"  MRR:              {sum(mrrs)/len(mrrs):.2f}")

    # --- Chunk coherence proxy ---
    print("\n--- Chunk coherence (avg pairwise sentence similarity within chunks) ---")
    embedder = get_embedder()
    sample_text = eval_set["documents"][0]["text"]
    sentences = [s.strip() for s in sample_text.split(".") if s.strip()]
    if len(sentences) >= 2:
        embeddings = embedder.encode(sentences, normalize_embeddings=True)
        score = chunk_coherence_score(list(embeddings))
        print(f"  coherence score (sample doc): {score:.3f}  (higher = more internally coherent)")

    # --- Performance ---
    print("\n--- Performance ---")
    print(f"  avg ingest latency: {sum(ingest_latencies)/len(ingest_latencies)*1000:.1f} ms/doc")
    sorted_q = sorted(query_latencies)
    p50 = sorted_q[len(sorted_q) // 2]
    p95 = sorted_q[min(len(sorted_q) - 1, int(len(sorted_q) * 0.95))]
    print(f"  query latency p50:  {p50*1000:.1f} ms")
    print(f"  query latency p95:  {p95*1000:.1f} ms")

    # --- Arabic diacritics comparison ---
    print("\n--- Arabic: diacritized vs. stripped retrieval ---")
    ar_doc = next(d for d in eval_set["documents"] if "ar_diacritized" in d["filename"])
    stripped_text = strip_diacritics(ar_doc["text"])
    stripped_path = _write_temp_doc(
        tmp_dir, ar_doc["filename"].replace("diacritized", "stripped"), stripped_text
    )
    ingest_file(stripped_path)

    ar_query = next(q for q in eval_set["queries"] if "معدل" in q["question"] or "أداء" in q["question"])
    results = retrieve(ar_query["question"], top_k=3)

    def _label(filename: str) -> str:
        if "diacritized" in filename:
            return "diacritized (original)"
        if "stripped" in filename:
            return "stripped (de-diacritized)"
        return "other document"

    for r in results:
        filename = r.metadata.get("filename") or ""
        print(f"  {_label(filename):24s} score={r.combined_score:.3f}  ({filename})")
    print(
        "  (Both variants are retrievable; this comparison quantifies whether "
        "stripping diacritics measurably changes ranking for this query set — "
        "run with more Arabic queries for a statistically meaningful gap.)"
    )


if __name__ == "__main__":
    print(f"Running benchmark suite against isolated store at {_BENCH_DIR}\n")
    run()
