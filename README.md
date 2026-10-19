# RegSearch

Search the US Code of Federal Regulations (CFR) in plain English and get back the
exact sections that answer your question, each with its full citation
(e.g. `4 CFR § 2.3`).

RegSearch pulls official regulation text from the public
[eCFR API](https://www.ecfr.gov/developers/documentation/api/v1), splits it into
section-level chunks that keep their place in the CFR hierarchy, and retrieves
them with a hybrid of semantic (vector) and keyword (BM25) search.

**Who it's for:** anyone who has to find "where does the regulation say that?":
compliance staff, lawyers and paralegals, researchers, journalists, or developers
building tools on top of federal regulations. Exact legal terms are often hard to
guess, so RegSearch matches on meaning as well as keywords.

> **Status:** early development. Ingestion, indexing, hybrid retrieval and an
> evaluation harness work today. The reranker, the cited-answer layer and the
> HTTP API are next. See [ROADMAP.md](ROADMAP.md).

## How it works

```
eCFR API ──► fetcher ──► XML parser ──► chunker ──► index (sqlite-vec + BM25) ──► hybrid search
 (XML)       (cached)    (hierarchy)    (JSONL)         single SQLite file          (RRF fusion)
```

- **`ingest/`**: downloads a full CFR title for a given date (cached under
  `data/raw/`), parses the `<DIV>` hierarchy (title › chapter › part › subpart ›
  section), and writes one chunk per section, splitting long sections at
  paragraph boundaries. Every chunk carries its citation and heading path.
- **`index/`**: embeds chunks with `BAAI/bge-small-en-v1.5`, with the heading
  path prepended for context, and stores the vectors (sqlite-vec), metadata and
  a BM25 index in one SQLite database.
- **`retrieval/`**: vector, BM25 and hybrid search. The hybrid mode combines
  the two rankings with reciprocal rank fusion (k = 60).
- **`eval/`**: a hand-written gold set of 47 questions (`eval/gold.jsonl`),
  including some with no answer in the corpus, plus a harness that reports
  recall@5, recall@10 and MRR@10 for each retrieval mode.

## Built with

Python 3.12 · [uv](https://docs.astral.sh/uv/) · sentence-transformers ·
sqlite-vec · rank-bm25 · requests · pytest · ruff · GitHub Actions.
FastAPI is planned for the API layer.

## Getting started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/afzal-xyz/regsearch.git
cd regsearch
uv sync
```

### 1. Fetch and chunk a CFR title

Title 4 (Accounts) is small and a good place to start:

```bash
uv run python -m ingest.chunker --title 4 --date 2024-07-18
```

This downloads `data/raw/title-4-2024-07-18.xml` if it isn't cached yet and
writes `data/chunks/title-4-2024-07-18.jsonl`.

### 2. Build the index

```bash
uv run python -c "from index.store import build_index; print(build_index('data/chunks/title-4-2024-07-18.jsonl'))"
```

The first run downloads the embedding model (~130 MB). The index is written to
`data/index/regsearch.db`.

### 3. Search

```bash
uv run python -m retrieval.search "when must a GAO employee file a financial disclosure" --k 5
```

### 4. Evaluate retrieval

```bash
uv run python -m eval.run            # all modes; --mode vector|bm25|hybrid
```

Results go to `eval/results/`. Current baseline on Title 4 (245 chunks, 42
answerable questions):

| Mode   | recall@5 | recall@10 | MRR@10 |
| ------ | -------- | --------- | ------ |
| vector | 0.81     | 0.90      | 0.66   |
| bm25   | 0.45     | 0.55      | 0.37   |
| hybrid | 0.69     | 0.76      | 0.54   |

Plain RRF currently scores *below* vector-only search, because the weaker BM25
ranking drags it down. Fixing that is the top item on the roadmap.

## Development

```bash
uv run pytest                 # full test suite
uv run pytest -m "not slow"   # offline tests only (what CI runs)
uv run ruff check .
```

CI (GitHub Actions) runs ruff and the offline tests on every push.

Raw eCFR downloads (`data/raw/`), chunks, indexes and eval outputs are
gitignored. Everything can be rebuilt with the commands above.

## Data

All regulation text comes from the public eCFR service run by the US Office of
the Federal Register. RegSearch does not handle any personal data. Its output is
not legal advice: always check the official text on
[ecfr.gov](https://www.ecfr.gov).

## License

[MIT](LICENSE)
