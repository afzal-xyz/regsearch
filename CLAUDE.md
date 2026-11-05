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

- `corpus.toml` — the declared corpus: every title/date the index is built from.
- `pipeline/` — manifest, integrity gate, onboarding orchestrator.
- `ingest/` — eCFR fetcher, XML parser, chunker.
- `index/` — vector + BM25 index building.
- `retrieval/` — hybrid retrieval, RRF fusion, `params.json` (tuned constants +
  provenance) and the cross-validated tuner.
- `rag/` — reranker and LLM answer layer.
- `eval/` — evaluation harness, `gold.jsonl`, and the gold-set gate/scaffolder.
- `api/` — FastAPI service.
- `tests/` — pytest suite.
- `data/raw/` — raw eCFR downloads (**gitignored, never commit**).

## Rules

- Every chunk must carry its full citation path (e.g. `40 CFR § 60.1`) as metadata.
- Never commit `data/raw/`.
- Run `pytest` before declaring a milestone done.
- Retrieval changes must be checked against `eval/gold.jsonl`.
- Add a title only via `python -m pipeline.onboard --title N --date D`; it is
  what keeps `corpus.toml`, the index and the gold set in agreement.
- Every pipeline stage operates on the whole declared corpus, never one title.
  Do not add per-title shortcuts that skip the others.
- A new title is not onboarded until the gold set covers it (12 prose, 4
  citation, 2 negatives). `--allow-gold-gaps` is for the authoring window only.
- Fusion/rerank constants are read from `retrieval/params.json`, never edited by
  hand: run `python -m retrieval.tune`. It adopts a change only when the
  cross-validated gain beats 1/n; an in-sample improvement alone is not evidence.
- Never report a tuned parameter set's in-sample score as its result.

## Commands

Managed with [uv](https://docs.astral.sh/uv/) — prefer `uv` over `pip`/`python` directly.

| Task | Command |
| --- | --- |
| Sync the venv | `uv sync` |
| Add a dependency | `uv add <package>` |
| Add a dev dependency | `uv add --dev <package>` |
| Run the API | `uv run uvicorn api.main:app --reload` |
| Run tests | `uv run pytest` |
| Onboard / refresh the corpus | `uv run python -m pipeline.onboard` |
| Add a title | `uv run python -m pipeline.onboard --title N --date YYYY-MM-DD` |
| Gate the corpus | `uv run python -m pipeline.integrity` |
| Gate the gold set | `uv run python -m eval.goldset --check` |
| Scaffold gold for a title | `uv run python -m eval.goldset --draft N` |
| Re-tune retrieval params | `uv run python -m retrieval.tune` (`--dry-run` to preview) |
