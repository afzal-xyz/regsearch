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
  provenance), the cross-validated tuner, and `prescore.py` (frozen candidates
  + cross-encoder scores, so the sandbox can rank without models).
- `rag/` — reranker and LLM answer layer.
- `eval/` — evaluation harness, `gold.jsonl`, and the gold-set gate/scaffolder.
- `api/` — FastAPI dashboard: `runs.py` (read-only view over `eval/results/`),
  `jobs.py` (allowlisted subprocess runner), `authoring.py` (draft queues, the
  lexical overlap meter, gold promotion), `sandbox.py` (cache-only parameter
  exploration), Jinja templates, vendored htmx.
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
- The dashboard never computes a benchmark number. It reads result files and
  shells out to the same CLIs CI runs (`api/jobs.py`); it must not grow an
  in-process eval path that could drift from the gate.
- No live retrieval preview beside gold-question authoring. Testing a question
  against the engine while writing it trains the author to phrase questions the
  current engine already answers, and the gold set stops measuring anything.
  The overlap meter is allowed because it is one-sided: driving it green can
  only remove lexical shortcuts, and no reading of it tells you retrieval
  succeeded, so there is nothing to overfit to.
- Authoring writes only to `eval/gold_titleN_draft.jsonl`. `eval/gold.jsonl` is
  touched solely by "Approve & merge", which validates against the live index
  and is all-or-nothing.
- Reading a draft queue never writes it. Hand-edited JSONL is normalised in
  memory and the repairs are reported; they reach disk only when the author
  saves a row. A GET must not churn a tracked file or fight someone's editor.
- "Could not check" is never rendered as a pass. An unindexed title, a missing
  section or an empty body puts the overlap meter in `UNKNOWN`, which reads
  "NOT CHECKED", blocks approval, and reports `distinctive: null` over JSON —
  a 0.0 there would read as clean.
- The meter's bands come from `python -m api.authoring --calibrate` over the
  approved gold set, with the corpus and gold digest recorded beside them.
  Bands that flag a large share of already-approved questions are wrong.
- No UI writes `retrieval/params.json`. The parameter sandbox is in-sample and
  exploratory; only `retrieval.tune` adopts, because only it cross-validates.
- The sandbox scores only from `retrieval/prescore.py`'s cache and is handed a
  reranker that raises. It must never gain the ability to embed, retrieve or
  cross-encode — that is what keeps a slider move milliseconds instead of
  minutes, and keeps torch out of the web process.
- Sandbox sliders span exactly `retrieval.tune.GRID`. Widening the search space
  means editing `GRID`, so the tuner can always re-check anything found there.

## Commands

Managed with [uv](https://docs.astral.sh/uv/) — prefer `uv` over `pip`/`python` directly.

| Task | Command |
| --- | --- |
| Sync the venv | `uv sync` |
| Add a dependency | `uv add <package>` |
| Add a dev dependency | `uv add --dev <package>` |
| Run the dashboard | `uv run uvicorn api.main:app --reload` |
| Run tests | `uv run pytest` |
| Onboard / refresh the corpus | `uv run python -m pipeline.onboard` |
| Add a title | `uv run python -m pipeline.onboard --title N --date YYYY-MM-DD` |
| Gate the corpus | `uv run python -m pipeline.integrity` |
| Gate the gold set | `uv run python -m eval.goldset --check` |
| Scaffold gold for a title | `uv run python -m eval.goldset --draft N` |
| Re-tune retrieval params | `uv run python -m retrieval.tune` (`--dry-run` to preview) |
| Calibrate the overlap meter | `uv run python -m api.authoring --calibrate` |
| Warm the sandbox cache | `uv run python -m retrieval.prescore` (`--no-rerank` for a fast, fusion-only cache) |
