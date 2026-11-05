# Roadmap

What's built and what's next. Every retrieval change gets measured against
`eval/gold.jsonl` before it lands.

## Done

- [x] eCFR fetcher with on-disk caching, streaming, rate limiting and XML validation
- [x] XML parser that follows the `TYPE` hierarchy (title › chapter › part › subpart › section)
- [x] Hierarchy-aware chunker with full citation and paragraph anchors
- [x] Embeddings (bge-small) + sqlite-vec store + persisted BM25 index
- [x] Hybrid retrieval with reciprocal rank fusion
- [x] Eval harness: 143-question gold set, recall@k / MRR, per-question detail
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

- [x] **Cross-encoder reranker** (`rag/reranker.py`) over hybrid's top-k.
      Hybrid found the answer in the top 10 for 93% of prose questions but
      ranked it no better than vector-only; the reranker converts that recall
      into rank — prose MRR 0.611 → 0.691 and recall@5 0.798 → 0.845 with
      recall@10 unchanged. Citation lookups are pinned and never reordered.
      Reranking *more* than k measured strictly worse, so the candidate pool
      is 10, not the conventional 30. See "Cross-encoder reranking" in the
      README.

- [x] **Formalized title onboarding** (`pipeline/`). `corpus.toml` declares the
      corpus; `pipeline.onboard` runs fetch → chunk → index → integrity → gold
      → eval → regression over **all** declared titles, so a title cannot be
      half-onboarded. Adds fetch retry with backoff (eCFR 504s on large
      titles), an integrity gate (chunk schema/citations, index-vs-manifest
      coverage, a retrieval self-lookup probe), a gold-set coverage gate with
      draft scaffolding, and gold-set fingerprinting so the regression gate
      does not fire on a changed question set. Validated by onboarding Title 1.
- [x] Corpus grown to Titles 1, 4 and 31 (6,564 chunks, 143 gold questions).
- [x] **Parameter tuning as a pipeline stage** (`retrieval/tune.py`,
      `retrieval/params.json`). Fusion and rerank constants carry provenance;
      the integrity gate warns when it goes stale; the `tune` stage grid-searches
      with 5-fold cross-validation and adopts only a gain beyond 1/n. First
      run: declined to change anything — the in-sample "win" was two questions
      worse held out. Adds a `preflight` stage. See "Parameters are re-tuned,
      honestly" in the README.

- [x] **Larger reranker.** `bge-reranker-base` replaces MiniLM-L-6 as the
      default: prose MRR 0.664 → 0.696 and recall@5 0.837 → 0.867 on identical
      candidates, at 5× the per-query latency on CPU. The reranker is now part
      of the tuning provenance, so swapping it forces a re-tune.
- [x] **CI gate on parameter staleness** (`retrieval.tune --check`): a title
      or gold-set change cannot merge without a matching `params.json`.

## Next: retrieval quality

- [ ] **Latency budget.** 8 s/query on CPU is fine for eval and batch, not for
      an interactive API. Decide GPU vs the small reranker vs a two-stage
      rerank (MiniLM over 30, bge over the top 10) once the API exists.
- [ ] **A better BM25 gate signal.** Score contrast barely separates "BM25
      found it" from "BM25 missed" (0.55 vs 0.44 mean confidence); a
      retriever-agreement factor is implemented as a tunable
      (`agreement_scale`) but did not win cross-validated. Candidates: query
      length / IDF mass, or a tiny learned gate once the gold set is larger.
      This is what would lift hybrid's Title 1 prose MRR; tuning cannot.
- [ ] Revisit stemming once the gold set is larger — Porter2 edged out the
      conservative S-stemmer by one question, but conflates applicant/applicable
      and administrator/administrative (see the README)
- [ ] Grow the gold set with harder paraphrase and prose negative questions
- [ ] Raise the per-title gold floor from 12/4/2 toward the 42/12 the first two
      titles carry, as authoring capacity allows

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
