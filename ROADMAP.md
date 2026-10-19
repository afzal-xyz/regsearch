# Roadmap

What's built and what's next. Every retrieval change gets measured against
`eval/gold.jsonl` before it lands.

## Done

- [x] eCFR fetcher with on-disk caching, streaming, rate limiting and XML validation
- [x] XML parser that follows the `TYPE` hierarchy (title › chapter › part › subpart › section)
- [x] Hierarchy-aware chunker with full citation and paragraph anchors
- [x] Embeddings (bge-small) + sqlite-vec store + persisted BM25 index
- [x] Hybrid retrieval with reciprocal rank fusion
- [x] Eval harness: 47-question gold set, recall@k / MRR, per-question detail
- [x] CI: ruff + offline pytest on GitHub Actions

## Next: retrieval quality

- [ ] **Make hybrid beat vector-only.** Plain RRF (recall@10 0.76) currently
      trails vector-only search (0.90). Try weighted RRF, score-normalized fusion,
      and better BM25 tokenization (stemming, keeping legal tokens like `§` and
      `(b)(1)`).
- [ ] Cross-encoder reranker (`rag/`) over the fused top-k
- [ ] Section-aware dedup: merge chunks that come from the same section in the results
- [ ] Grow the gold set beyond Title 4 and add harder paraphrase and negative questions
- [ ] Track eval results over time (compare against the last run, flag regressions)

## Next: answers

- [ ] LLM answer layer that must cite CFR sections for every claim
- [ ] Citation checker: reject answers that cite sections missing from the retrieved context
- [ ] "Not found in the regulations" path for negative questions

## Next: product

- [ ] `python -m index.store` CLI for building the index (replaces the `python -c` one-liner)
- [ ] One command for the whole pipeline (fetch → chunk → index) across several titles
- [ ] FastAPI service (`api/`): `/search` and `/answer` endpoints
- [ ] Minimal web UI with results that link to ecfr.gov
- [ ] Optional pgvector backend for larger corpora
- [ ] Ingest all 50 titles; incremental updates using eCFR's change dates
- [ ] Point-in-time search ("what did this section say on date X?")
