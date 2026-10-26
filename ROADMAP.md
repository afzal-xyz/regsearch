# Roadmap

What's built and what's next. Every retrieval change gets measured against
`eval/gold.jsonl` before it lands.

## Done

- [x] eCFR fetcher with on-disk caching, streaming, rate limiting and XML validation
- [x] XML parser that follows the `TYPE` hierarchy (title › chapter › part › subpart › section)
- [x] Hierarchy-aware chunker with full citation and paragraph anchors
- [x] Embeddings (bge-small) + sqlite-vec store + persisted BM25 index
- [x] Hybrid retrieval with reciprocal rank fusion
- [x] Eval harness: 61-question gold set, recall@k / MRR, per-question detail
- [x] CI: ruff + offline pytest on GitHub Actions
- [x] **Make hybrid beat vector-only.** Section-level fusion + confidence-weighted
      BM25 took recall@10 from 0.76 to 0.93, past vector-only's 0.90. See
      "Hybrid fusion" in the README.
- [x] Section-aware dedup: hybrid now fuses over sections and returns one result each
- [x] **Citation-aware BM25 tokenizer + stemming.** Keeps `§ 2.3(b)(1)` and
      `4 CFR 25.14` whole and normalizes every spelling onto one token; indexes
      the citation and heading path, not just the body. Retrieval by citation
      went from 0.01 to 1.00 hit@1, and BM25 recall@10 from 0.55 to 0.69.
- [x] **Expand section ranges.** `§§ 28.18 through 28.21` now indexes every
      section it covers, not just the endpoints, capped at 100 sections. A
      lookup for an interior section finds the text governing it; before, it
      never did.
- [x] Gold set split by question `kind`, with 6 citation lookups added — two
      of which target the middle of a range. `eval/run.py` reports prose and
      citation metrics separately, since blending them hides both.
- [x] **Citation override only for sections that exist.** Naming a
      non-existent section (`4 CFR 4.999`) or one from another title
      (`31 CFR 1010.230`) used to force full BM25 trust on the leftover prose,
      pulling keyword noise into the top 5. Both are now negative gold
      questions, and the override checks the index first.
- [x] Citation-lookup set grown from 6 to 12 (three target the middle of a
      range), plus the 2 citation-form negatives above. Hybrid: recall@5 0.92,
      recall@10 1.00, MRR 0.83; prose metrics unchanged.
- [x] **Eval run comparison + regression gate + report** (`eval/compare.py`):
      diffs the latest run against the previous or a pinned baseline per mode
      and question kind, fails CI when a metric drops more than the threshold
      (adaptive 1.5/n by default, `--threshold` for flat), and renders
      `eval/report.html` — delta tables plus full history with git commits and
      trend sparklines. `eval/run.py` stamps each run with its commit and
      refreshes the page.

## Next: retrieval quality

- [ ] Cross-encoder reranker (`rag/`) over the fused top-k
- [ ] **Re-tune the BM25 confidence thresholds on a larger gold set.** They sit
      on a wide plateau on Title 4, but 54 answerable questions over one small
      title is thin evidence that the chosen contrast band transfers. They are
      calibrated per `ANALYZER_VERSION` and must be re-checked with it.
- [ ] Revisit stemming once the gold set is larger — Porter2 edged out the
      conservative S-stemmer by one question, but conflates applicant/applicable
      and administrator/administrative (see the README)
- [ ] Grow the gold set beyond Title 4 and add harder paraphrase and prose negative questions

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
