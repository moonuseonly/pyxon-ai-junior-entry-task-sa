# Arabic-Aware Document Intelligence for RAG

An AI-powered document parser that ingests PDFs, DOCX, and TXT files, understands
their structure, chunks them intelligently, and prepares them for retrieval-augmented
generation — with first-class support for Arabic text, including diacritics (harakat/tashkeel).

Most RAG pipelines are built and benchmarked on English text and silently strip or
mangle Arabic diacritics during preprocessing. This project treats Arabic as a
first-class citizen: normalization is diacritics-preserving by default, and the
benchmark suite explicitly measures retrieval quality on diacritized text.

## Why this exists

Document parsing is the part of a RAG system everyone skips past to get to the "fun"
LLM part — but retrieval quality is bottlenecked by chunk quality, and chunk quality
is bottlenecked by parsing. This project focuses on that unglamorous middle layer:

- **Format-agnostic ingestion** — PDF, DOCX, and TXT through one interface
- **Strategy-aware chunking** — the system inspects a document's structure and
  picks between fixed-size and structure-aware dynamic chunking, rather than
  applying one strategy to every document
- **Dual storage** — embeddings go to a vector store (Chroma) for semantic search,
  structured metadata goes to a relational store (Postgres/SQLite) for filtering
  and analytics queries that vector search alone can't answer
- **Arabic-first normalization** — configurable diacritic handling instead of the
  usual silent stripping, plus bidirectional text handling
- **A real benchmark suite** — retrieval precision/recall, chunk coherence, and
  latency, not just "it runs"

## Architecture

```
                 ┌─────────────┐
   PDF/DOCX/TXT →│   Parsers   │
                 └──────┬──────┘
                        ▼
                 ┌─────────────────┐
                 │ Document         │  structure & language detection,
                 │ Analyzer         │  decides fixed vs. dynamic strategy
                 └──────┬───────────┘
                        ▼
                 ┌─────────────┐
                 │  Chunker    │  fixed-size  |  structure-aware dynamic
                 └──────┬──────┘
                        ▼
             ┌──────────┴───────────┐
             ▼                      ▼
     ┌───────────────┐     ┌────────────────┐
     │ Vector Store   │     │  SQL Store      │
     │ (Chroma)       │     │  (Postgres/     │
     │ embeddings     │     │   SQLite)       │
     └───────┬────────┘     │  metadata,      │
             │              │  chunk records  │
             │              └────────┬────────┘
             ▼                       │
     ┌────────────────────┐         │
     │  Hybrid Retriever   │◄────────┘
     │  (vector + keyword) │
     └──────────┬──────────┘
                ▼
        ┌───────────────┐
        │  RAG API        │  FastAPI: /ingest, /query, /documents
        └───────────────┘
```

## Chunking strategy selection

The analyzer looks at heading density, paragraph-length variance, and detected
section markers to classify a document as **uniform** (e.g. a form, a structured
report with consistent section lengths) or **structured/mixed** (e.g. a book with
chapters, a document mixing prose and tables). Uniform documents get fast fixed-size
chunking with overlap; structured documents get dynamic chunking that respects
detected boundaries so a chunk doesn't cut a section in half. See
`app/processing/document_analyzer.py` for the heuristics and
`app/processing/chunking.py` for both strategies.

## Arabic handling

`app/processing/arabic_utils.py` implements:
- Unicode-range-based diacritic (tashkeel) detection and optional stripping —
  **preserving by default**, since removing diacritics can change meaning
- Arabic-Indic digit normalization
- Tatweel (kashida) removal (a common OCR/typing artifact, not semantic content)
- Mixed-direction (Arabic/Latin) text-safe chunk boundaries, so a chunk doesn't
  split mid-word across a directionality change

## Running it

```bash
docker compose up --build
```

This starts the API (`localhost:8000`) and a Postgres instance. Chroma persists to
a local volume. Interactive API docs: `localhost:8000/docs`.

> **First run note:** the embedding model (`paraphrase-multilingual-mpnet-base-v2`)
> downloads from Hugging Face on first startup — expect a short delay and make
> sure the machine running this has normal internet access. After the first run
> it's cached and startup is fast.

Without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Try it

```bash
# Ingest a document
curl -X POST localhost:8000/ingest -F "file=@sample.pdf"

# Query
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "top_k": 5}'
```

## Benchmarks

```bash
python -m benchmarks.run_benchmark
```

Runs against the small labeled eval set in `benchmarks/data/` (English + Arabic
documents with hand-written queries and relevance judgments) and reports:

- **Retrieval quality**: precision@k, recall@k, MRR
- **Chunking quality**: a coherence proxy (average intra-chunk sentence-embedding
  similarity vs. cross-chunk-boundary similarity — good chunks should score higher
  internally than at their edges)
- **Performance**: ingestion throughput, query latency (p50/p95)
- **Arabic-specific**: retrieval quality on diacritized vs. de-diacritized copies
  of the same source, to quantify what stripping diacritics costs

## Project status / roadmap

This implementation covers the core pipeline solidly (parsing, strategy-aware
chunking, dual storage, hybrid retrieval, Arabic normalization, benchmarks). It
does **not** yet implement Graph RAG or RAPTOR-style hierarchical summarization —
both are natural next layers on top of this chunk store and are noted here rather
than half-implemented:

- [ ] RAPTOR-style recursive summarization for multi-level retrieval
- [ ] Graph RAG layer over extracted entities/relations
- [ ] Swap-in for a dedicated Arabic embedding model (e.g. AraBERT/CAMeLBERT) behind
      the same interface, benchmarked against the current multilingual model

## Tech stack

FastAPI · LangChain-style modular pipeline (no framework lock-in) ·
`sentence-transformers` (multilingual embeddings) · Chroma · SQLAlchemy +
PostgreSQL/SQLite · `pdfplumber` (font-size-based heading detection), `python-docx` ·
Docker / Docker Compose

## License

MIT
