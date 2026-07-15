# CLAUDE.md

**RegSearch** — hybrid RAG search over the US Code of Federal Regulations (eCFR).

## Stack

Python 3.12, uv, FastAPI, sentence-transformers, pgvector (or sqlite-vec), rank-bm25.

## Architecture

- **ingest** — eCFR API fetcher → XML parser → hierarchy-aware chunker.
- **hybrid retrieval** — vector + BM25 with reciprocal rank fusion.
- **reranker** — cross-encoder reranker.
- **answer layer** — LLM answer layer with mandatory CFR citations.

## Layout

- `ingest/` — eCFR fetcher, XML parser, chunker.
- `index/` — vector + BM25 index building.
- `retrieval/` — hybrid retrieval and reciprocal rank fusion.
- `rag/` — reranker and LLM answer layer.
- `eval/` — evaluation harness and `gold.jsonl`.
- `api/` — FastAPI service.
- `tests/` — pytest suite.
- `data/raw/` — raw eCFR downloads (**gitignored, never commit**).

## Rules

- Every chunk must carry its full citation path (e.g. `40 CFR § 60.1`) as metadata.
- Never commit `data/raw/`.
- Run `pytest` before declaring a milestone done.
- Retrieval changes must be checked against `eval/gold.jsonl` once it exists.

## Commands

Managed with [uv](https://docs.astral.sh/uv/) — prefer `uv` over `pip`/`python` directly.

| Task | Command |
| --- | --- |
| Sync the venv | `uv sync` |
| Add a dependency | `uv add <package>` |
| Add a dev dependency | `uv add --dev <package>` |
| Run the API | `uv run uvicorn api.main:app --reload` |
| Run tests | `uv run pytest` |
