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

> **Status:** early development. A one-command onboarding pipeline, hybrid
> retrieval, a cross-encoder reranker and a gated evaluation harness work
> today. The cited-answer layer and the HTTP API are next. See [ROADMAP.md](ROADMAP.md).

## How it works

```
eCFR API ─► fetcher ─► XML parser ─► chunker ─► index (sqlite-vec + BM25) ─► hybrid search ─► reranker
 (XML)      (cached)   (hierarchy)   (JSONL)        single SQLite file         (RRF fusion)   (cross-encoder)

            └────────────── pipeline/onboard.py drives all of this, per corpus.toml ──────────────┘
                              fetch → chunk → index → integrity → gold → eval → regression
```

- **`ingest/`**: downloads a full CFR title for a given date (cached under
  `data/raw/`), parses the `<DIV>` hierarchy (title › chapter › part › subpart ›
  section), and writes one chunk per section, splitting long sections at
  paragraph boundaries. Every chunk carries its citation and heading path.
- **`index/`**: embeds chunks with `BAAI/bge-small-en-v1.5`, with the heading
  path prepended for context, and stores the vectors (sqlite-vec), metadata and
  a BM25 index in one SQLite database. BM25 uses a citation-aware analyzer
  (see [BM25 tokenization](#bm25-tokenization)).
- **`retrieval/`**: vector, BM25 and hybrid search. The hybrid mode fuses the
  two rankings over *sections*, weighting BM25 by how much it can be trusted on
  that particular query (see [Hybrid fusion](#hybrid-fusion)).
- **`rag/`**: a cross-encoder reranker that re-reads the question against each
  of hybrid's candidates and reorders them (see
  [Cross-encoder reranking](#cross-encoder-reranking)).
- **`pipeline/`**: the corpus manifest (`corpus.toml`), the integrity gate and
  the onboarding orchestrator (see [Onboarding a title](#onboarding-a-title)).
- **`eval/`**: a hand-written gold set of 143 questions (`eval/gold.jsonl`) —
  prose questions, citation lookups, and some with no answer in the corpus —
  plus a harness that reports recall@5, recall@10 and MRR@10 per retrieval
  mode, broken out by question `kind`. `eval/compare.py` diffs runs against
  the previous one or a pinned baseline, fails on regressions beyond a
  threshold, and renders `eval/report.html` with the full run history.

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

### 1. Build the corpus

`corpus.toml` declares which CFR titles the index is built from. One command
fetches, chunks, indexes, verifies and evaluates all of them:

```bash
uv run python -m pipeline.onboard
```

The first run downloads the title XML (~12 MB for the three declared titles)
and the embedding model (~130 MB), and writes `data/index/regsearch.db`.
Later runs skip whatever is already current, so this is also the right command
after changing the chunker or the BM25 analyzer.

To add a title, see [Onboarding a title](#onboarding-a-title). The individual
stages are still available on their own (`python -m ingest.chunker --title 4
--date 2024-07-18`) when you want to poke at one.

### 2. Search

```bash
uv run python -m retrieval.search "when must a GAO employee file a financial disclosure" --k 5
```

Add the cross-encoder reranker on top (first run downloads a ~90 MB model):

```bash
uv run python -m rag.reranker "when must a GAO employee file a financial disclosure" --k 5
```

### 3. Evaluate retrieval

```bash
uv run python -m eval.run            # all modes; --mode vector|bm25|hybrid|rerank
```

Results go to `eval/results/` (one JSON per run, stamped with the git commit
it ran on). Current baseline on Titles 1, 4 and 31 (6,564 chunks, 143
questions: 127 answerable, 16 with no answer in the corpus).

The answerable questions are two kinds, reported separately because they are
different tasks: 98 phrased in everyday language, and 29 known-item lookups that
name a section outright. Negatives are excluded from recall/MRR; their rankings
are recorded per question in the results JSON.

**Prose questions** (n = 98):

| Mode   | recall@5 | recall@10 | MRR@10   |
| ------ | -------- | --------- | -------- |
| vector | 0.806    | 0.898     | 0.622    |
| bm25   | 0.571    | 0.633     | 0.440    |
| hybrid | 0.816    | 0.908     | 0.607    |
| rerank | **0.898**| **0.949** | **0.718**|

Hybrid finds the answer within 10 for 91% of prose questions but *ranks* it
worse than vector-only does (MRR 0.607 vs 0.622). The reranker is what
converts recall into rank — and with `bge-reranker-base` reading 12
candidates rather than 10, it now also *adds* recall: 95% of prose answers in
the top 10, 90% in the top 5, MRR 0.718. See
[Cross-encoder reranking](#cross-encoder-reranking).

Adding Title 1 moved these numbers, and the direction is worth recording.
Its 14 prose questions are drawn from short procedural sections about Federal
Register publishing, which is bad terrain for BM25 — its prose recall@10 fell
0.702 → 0.633 — and hybrid inherited some of that, slipping just below
vector-only on MRR where it had been at parity. The reranker absorbed it:
rerank's prose recall@5 rose 0.798 → 0.837 across the same change. These are
not comparable deltas in the regression-gate sense (different question sets,
see [Onboarding a title](#onboarding-a-title)); they are the new baseline.

**Citation lookups** (n = 29):

| Mode   | recall@5 | recall@10 | MRR@10   |
| ------ | -------- | --------- | -------- |
| vector | 0.448    | 0.483     | 0.400    |
| bm25   | 0.966    | 0.966     | 0.888    |
| hybrid | 0.966    | **1.000** | 0.929    |
| rerank | **1.000**| **1.000** | **0.940**|

Hybrid finds every cited section within 10 and puts most at rank 1 — Title 1's
5 new lookups landed cleanly. The remaining loss is the mid-range lookups, where the answer is a section that
*covers* the cited one via a `§§ X through Y` range rather than naming it (see
[Section ranges](#section-ranges)). Rerank is flat here by design — citation
hits are pinned and never reordered.

Vector search is near-useless here — embeddings have no notion of a citation —
which is why BM25 carries this case. See
[BM25 tokenization](#bm25-tokenization).

### 4. Compare runs and catch regressions

```bash
uv run python -m eval.compare                          # latest vs previous run
uv run python -m eval.compare --against 20261025-131252.json   # vs a pinned baseline
uv run python -m eval.compare --threshold 0.05         # flat threshold
uv run python -m eval.compare --no-fail                # warn-only (never exit 1)
```

Prints every metric's change per mode and per question kind, flags any drop,
and exits non-zero when a drop exceeds the threshold — so CI can run it after
`eval.run` and block a retrieval regression. It also (re)writes
**`eval/report.html`**, a self-contained page with the latest-vs-reference
delta tables, a verdict line, and the full run history — every run's metrics,
its git commit, and recall@10 trend sparklines per kind. `eval/run.py`
refreshes the page after every run, so opening it always shows current state.

**The default threshold is adaptive: a metric may drop by at most 1.5/n for
its own group.** These metrics move in steps of one question (1/n): on the 42
prose questions one flipped question moves recall by 0.024, while on the 12
citation lookups it moves it by 0.083. A single flat number either ignores
real two-question losses in the big group or cries wolf on one-question noise
in the small one. 1.5/n sits between one question and two — a lone borderline
question re-ranking is noise on a hand-written gold set this size, a second
one is a pattern — and it scales automatically as the gold set grows (the
slack tightens, which is what more evidence should do). Pass `--threshold` for
a flat absolute limit when comparing across gold-set changes, where n itself
moved.

## Hybrid fusion

Hybrid search used to score *below* vector-only search (recall@10 0.76 vs 0.90).
Plain RRF over chunks had three problems, each fixed in `retrieval/search.py`:

**1. Fusing chunks instead of sections.** A section split into several chunks
occupied several RRF slots and crowded out whole other sections. Collapsing each
retriever's ranking to its best chunk per section first: 0.76 → 0.83.

**2. BM25 was trusted equally on every query.** BM25 always returns *something*.
Ask *"can I bring a gun into the GAO building?"* and it returns 50 confident-looking
chunks, none of which contain "gun" — the regulation says "firearms". Fusing that
noise at full strength pushed the correct section (`4 CFR § 25.14`, vector's rank 1)
out of the top 10 entirely.

The fix weights BM25 by the *contrast* of its own score distribution — the top
score over the score at rank 10. A real lexical match towers over the rest; noise
comes back flat. On the gold set this separates cleanly: the queries BM25 answers
well score 1.9–3.8, the ones where it returns garbage score 1.2–1.5. The weight
ramps linearly across that gap, so hybrid falls back to pure semantic order when
BM25 has nothing to offer. The measure is scale-free, so it does not move with
query length or corpus size.

**3. `rrf_k = 60` flattened the ranks.** With a 50-deep pool, ranks 1 and 50 differ
by less than 2×, so merely appearing in both pools (20% of this corpus) outweighed
being the single best semantic match. Lowered to 10.

Together these take recall@10 to 0.93, above vector-only's 0.90 and at the ceiling
of 0.93 set by the union of what the two retrievers can find at all. Recall@5 also
improves (0.81 → 0.83) and MRR holds at parity; raising MRR is the reranker's
job — see [Cross-encoder reranking](#cross-encoder-reranking).

Two caveats worth knowing: the gains hold across a wide parameter plateau
(`rrf_k` 5–60, pools 30–150, contrast thresholds 1.4–1.7), so they come from the
structure rather than tuned constants — but 54 answerable questions over one small
title is a thin basis for the thresholds, which should be re-tuned as the gold set
grows. And because vector dominates BM25 so heavily on this corpus, hybrid is
deliberately skewed semantic; a corpus with more exact-term queries would want the
BM25 weight raised.

## Onboarding a title

Adding a CFR title is one command:

```bash
uv run python -m pipeline.onboard --title 1 --date 2024-07-18 --name "General Provisions"
```

It declares the title in `corpus.toml`, then runs nine stages — `preflight`,
`fetch`, `chunk`, `index`, `integrity`, `gold`, `tune`, `eval`, `regression` —
and stops at the first failure rather than carrying a broken corpus forward.
`preflight` fails fast on what needs no download (manifest, gold-file schema,
both models load, params file readable), so a typo is found in seconds rather
than after a ten-minute index build. `tune` is described below.

**Every stage runs over the whole declared corpus, not just the new title.**
That is the point. A per-title pipeline produces a corpus that is correct
title-by-title and broken as a whole: an index holding a title the manifest
forgot, a gold set covering two titles out of three, chunks written by a newer
parser than the ones beside them. None of those raise an error anywhere —
every individual command succeeded. So the unit of work is the corpus,
`corpus.toml` is its definition, and a stage either brings all of it up to
date or fails.

Re-walking finished titles is cheap because every stage is idempotent and
skips work it does not need (fetch hits the cache, chunk and index compare
mtimes). Bringing an already-current three-title corpus up to date takes about
a second before the eval.

```bash
uv run python -m pipeline.manifest        # what is declared, and what exists on disk
uv run python -m pipeline.onboard --only integrity
uv run python -m pipeline.onboard --from index
uv run python -m pipeline.onboard --allow-gold-gaps   # while writing gold questions
```

### The integrity gate

Every check in `pipeline/integrity.py` exists because the failure it catches
is *silent* — the pipeline completes, the index builds, search returns
results, and the results are quietly wrong. It runs in three layers:

| Layer | Catches |
| --- | --- |
| chunk | missing fields, empty text, malformed citations, a chunk whose title or date disagrees with the manifest, duplicate `chunk_id`s (which `UNIQUE` would silently drop) |
| index | a declared title missing from the index, an **undeclared** title present in it, a chunk count that disagrees with the chunk file, missing vectors, an empty or stale BM25 index |
| retrieval | the index not actually retrieving: a sample of sections must come back rank 1 for their own citation |

That last layer matters because every structural check can pass on an index
that finds nothing. It is also the cheapest query whose right answer is known
without a gold set: ask for `4 CFR § 7.3`, get `4 CFR § 7.3`.

The gate caught two of its own bugs on first contact. It rejected 358 chunks
as "malformed citations" that turned out to be appendices — `4 CFR Appendix I
to Part 83` is a real citation path, and appendices are often where a rule's
tables and dollar thresholds live. The container after "to" is now
deliberately not enumerated, because every list of containers is a list of the
ones seen so far: the first draft allowed only `Part N` and failed on Title 4,
the second added `Subpart X of Part N` and failed on `Appendix A to Chapter V`.

### Gold-set coverage is a gate

Indexing a title without extending the gold set means the new text is searched
but never measured: the index grows, every metric stays flat because it is
averaged over questions that do not touch the new title, and a regression
there is invisible. So `eval/goldset.py` gates per-title coverage — 12 prose
questions, 4 citation lookups, and 2 negatives per declared title.

Twelve is the floor because `eval/compare.py` gates regressions at 1.5/n.
Below about a dozen questions a single flipped question moves recall by more
than 8 points, so the per-title numbers cannot distinguish a real regression
from a coin flip and the gate would either never fire or fire constantly.

`--draft` scaffolds a new title, and it is explicit about what a machine can
and cannot write:

```bash
uv run python -m eval.goldset --draft 1 --n-prose 14 --n-citation 5
```

- **Citation lookups are generated** — "What does `1 CFR § 1.1` provide?" is
  correct by construction, since the section named is the section wanted.
- **Negatives are generated** — a citation-form negative names a section
  number verified absent from the corpus, so its empty answer is correct too.
- **Prose questions are not generated.** The whole point of a prose question
  is that it asks in everyday words the regulation does not use. A question
  generated *from* the section body necessarily reuses that body's vocabulary,
  so it would test string matching, score near-perfectly, and measure nothing.
  The tool emits stratified **stubs** with the section text for a human to
  write against; `question` is left empty and the validator errors on an empty
  question, so a stub can never drift into the measured set.

### Negatives expire

A negative is only negative with respect to a corpus. *"When is the federal
income tax filing deadline?"* has no answer in Titles 4 and 31 and a perfectly
good one the moment Title 26 is onboarded — at which point the system is
penalised for answering it *correctly*, and nothing about the entry changed.

So each negative records the corpus it was checked against:

```json
{"question": "...", "relevant_citations": [], "negative_corpus": ["4@2024-07-18", "31@2024-07-18"]}
```

Growing the corpus past that marks it stale and the gate says which titles
need re-reading. A citation-form negative naming a section that *is* now
indexed is not merely stale but wrong, and that is a hard error.

### Regression runs know when not to fire

Onboarding a title adds its questions, so the first run afterwards averages
over a different population than the one before it. "Hybrid prose MRR fell
0.016" is then not a regression — it is a different measurement wearing the
same name. Each results file records a fingerprint of its question set
(count plus a digest over questions, expected citations and kinds), and
`eval/compare.py` detects the change, prints the deltas for information, and
declines to fail on them:

```
NOTE: the question set changed (122 -> 143 questions) between these two runs,
so the deltas above compare means taken over different question sets.
They are shown for information; the gate is not failing on them.
```

This matters more than it looks. A gate that cries wolf every time the gold
set grows is a gate people learn to pass `--no-fail` to, and then it is not a
gate at all.

### Parameters are re-tuned, honestly

The fusion constants in `retrieval/search.py` — the RRF constant, how much to
trust BM25, the confidence band that decides when BM25 is noise — are
calibrated against a score *distribution*, and that distribution moves every
time the corpus does. The code used to say "re-tune when the corpus grows".
When Title 1 was onboarded nobody did, and hybrid's prose MRR slipped below
plain vector search (0.595 vs 0.622) without any stage noticing. A comment is
not a mechanism, so now there is one:

- Parameters live in **`retrieval/params.json`** with a record of what they
  were tuned on (corpus, gold-set fingerprint, BM25 analyzer version).
- The **integrity gate warns** when that record no longer matches the live
  corpus, naming what changed.
- The **`tune` stage** (`retrieval/tune.py`) runs before `eval`, so the
  numbers recorded are for the parameters actually in use. It fetches each
  gold question's candidates once and scores a few thousand parameter sets
  in about a minute.
- **CI refuses a stale params file.** `python -m retrieval.tune --check`
  compares the committed `params.json` provenance against the committed
  `corpus.toml`, `gold.jsonl`, analyzer version and default reranker — all
  derivable from files, no index or model needed — and exits non-zero on any
  mismatch. Add a title or edit a question without re-tuning and the build
  fails with the reason:

  ```
  STALE: retrieval/params.json was not tuned for this corpus/gold set:
    - gold set changed: tuned on 143 questions, now 144
  run `uv run python -m pipeline.onboard --from tune` locally and commit retrieval/params.json
  ```

  This closes the one gap the pipeline had: it re-tuned on any machine with
  an index, but nothing stopped a change from being *merged* without one.

**What stops it from fooling itself.** "Pick the parameters that score best on
the gold set, then report the gold-set score" is circular — the number always
looks good and says nothing about the next title. So the tuner splits the
questions into five folds, picks the best parameters on four, scores them on
the fifth, and averages those held-out scores. A change is adopted only when
that honest estimate beats the current parameters by more than one question's
worth (1/n ≈ 0.008 here). Otherwise it keeps the current parameters and
records "checked against this corpus, still best" — which is exactly what the
staleness warning was asking for.

Here is what that protection did on the first run, against the real
(reranked) output on 127 questions:

| | current | best in-sample |
| --- | --- | --- |
| objective, in-sample | 0.827 | **0.834** |
| objective, cross-validated | **0.827** | 0.812 |

The in-sample "winner" would have shipped under a naive tuner. Held out, it is
two questions *worse*. Across three runs with the small reranker (hybrid
output, reranked output, and hybrid with two new hypotheses in the grid — a
retriever-agreement gate and very small RRF constants) the picture was the
same: in-sample gains of +0.01–0.04 in prose MRR, cross-validated gains of
+0.0007 to +0.003, all inside noise. The tuner correctly declined to move.

Then the reranker was swapped for `bge-reranker-base`, the provenance went
stale, and the same procedure **adopted** a change — 12 candidates instead of
10 and the agreement gate on — with a held-out gain of +0.010 against the
0.008 margin. Both verdicts came from the same rule. That is what makes it
trustworthy: it declines when the evidence is noise and moves when it is not,
and the reader does not have to take anyone's word for which is which.

**So why did hybrid dip, and does it matter?** The dip is specific to Title
1's prose, where BM25 is nearly blind: its MRR there is 0.107 against
vector's 0.681. In 11 of the 14 questions where hybrid ranks the answer below
vector, BM25 does not have the answer *anywhere* in its top 10, yet its
confidence gate lets it in at 30–60%, and under RRF a single co-occurring
section is enough to nudge vector's rank-1 answer to rank 2 or 3 (the gap
between RRF rank 1 and rank 2 at k=10 is 0.0076 — smaller than almost any
BM25 contribution). No threshold fixes a gate whose signal barely separates
"BM25 found it" from "BM25 missed" (mean confidence 0.55 vs 0.44). It does
not reach the user, though: hybrid's job in this pipeline is *recall* for the
reranker, and at that it still edges vector (recall@10 0.908 vs 0.898); the
reranker then reorders, and the final prose MRR is 0.664 — above vector's
0.622. The dip is real, it is in an intermediate stage, and it is not fixable
by turning knobs. What would fix it is a better gate signal or a stronger
reranker — see the limitations below.

### Do we need both a bi-encoder and a cross-encoder?

Yes, and BM25 as well — they do three different jobs, and none of them can do
another's:

- **The bi-encoder** (`bge-small`, the embedding model) turns every chunk into
  a vector *once, at index time*, and the question into a vector at query
  time. Finding the nearest vectors among 6,564 chunks takes milliseconds.
  Its weakness is that it never sees the question and a passage *together*,
  so it cannot notice that a passage uses all the right words to answer a
  different question. It is the only thing that can search the whole corpus
  quickly.
- **BM25** matches exact terms. It is what makes `4 CFR § 25.14` findable by
  its number — the bi-encoder has no notion of a citation — and it carries
  the citation lookups at 0.966 recall@5 where vector manages 0.448. On
  everyday-language questions it is the weakest of the three.
- **The cross-encoder** reads the question and one passage as a single input
  and scores the pair. It is far more accurate and far too slow for a corpus
  (one forward pass per passage), so it only ever sees the ten candidates the
  other two narrowed to. It is what turns recall into rank.

Dropping any one loses something measurable: no bi-encoder and there is no
fast recall; no BM25 and citation lookups collapse; no cross-encoder and prose
MRR falls from 0.664 to 0.595–0.622. "Something else" worth trying is a
*stronger* cross-encoder, not a different architecture.

### Limitations of the tuning safety net

Stated plainly, because a safety net people over-trust is worse than none:

1. **It can only guarantee "no worse on questions like these".** Five-fold
   cross-validation estimates how parameters generalise to *held-out
   questions from the same gold set*. A new title whose questions look
   nothing like the existing ones is a distribution shift the folds cannot
   see. Title 1 is the example: its prose questions are short, procedural,
   and de-vocabbed, and no tuning on Titles 4 and 31 would have predicted
   BM25 going blind on them.
2. **It cannot raise a number that is not a tuning problem.** The Title 1 dip
   is a *signal* problem (the confidence gate's input is weak), and the tuner
   proved that by finding nothing to adopt. Fixing it means a design change,
   measured the same way.
3. **143 questions is a small gold set.** One question moves recall by 0.7–1
   points; the noise margin is set accordingly, which means a real but small
   improvement (under ~1%) is invisible to the gate by construction. More
   questions per title tighten this — the ROADMAP tracks raising the floor.
4. **Negatives are not in the objective.** There is no rank to score when the
   right answer is "nothing", so the 16 negatives do not constrain tuning. A
   parameter set that happens to make the system more confidently wrong on
   negatives would not be caught here; their per-question rankings are
   recorded in the results JSON for inspection, not gated.
5. **The models are not tuned.** Which bi-encoder and which cross-encoder are
   deliberate experiments (a 130 MB download and a full re-index each), not
   knobs the pipeline turns on its own. A stronger cross-encoder is the most
   promising untested change.
6. **The objective is a choice.** Half recall@10, half MRR@10 — recall
   because it is the ceiling everything downstream can reach, MRR because it
   was the stuck metric. A different weighting would pick differently on the
   margin; it would not change the "within noise" verdicts above.

### Fetching is retried

eCFR renders a full title on demand and sheds load under it: a 504 on the
first attempt at a large title is routine, and the identical request succeeds
minutes later — Title 31 did exactly this. Transient failures (timeouts,
dropped connections, 429, 5xx) are retried with exponential backoff honouring
`Retry-After`. A 404 is not retried: that title/date will never exist, and
retrying only delays the error.

## Cross-encoder reranking

Hybrid search had a lopsided scorecard: it found the right section inside the
top 10 for **93%** of prose questions but its MRR was **0.611**, no better
than vector-only. The answer was in the list and in the wrong place — 18
questions had their answer at rank 2–10. That is a *ranking* problem, and a
reranker is the tool for it.

**Why hybrid can't fix this itself.** Both retrievers score the query and the
passage *separately*. The query becomes one vector; every chunk became a
vector at index time; the ranking is their similarity. That independence is
what makes searching 6,262 chunks fast, and it is also the ceiling: the model
never sees the question and the passage side by side, so it cannot notice that
a passage uses all the right words to answer a *different* question.

A cross-encoder does see them together. It takes `(question, passage)` as one
input and returns a single relevance score. One forward pass per passage makes
it hopeless over a corpus — but over the handful hybrid already narrowed to,
it is cheap, and it judges relevance far better.

Two decisions carried the result:

**1. Score the contextualized chunk, not the bare body.** CFR chunk bodies
often read as fragments — *"(a) The term means …"* — with nothing saying which
section they are from. The reranker is handed the same form the embedder saw:
heading path, newline, text. Without the heading path a cross-encoder is
guessing at what it is reading.

**2. Never rerank a citation lookup.** A question naming `§ 25.14` is a
known-item lookup that hybrid already resolved with certainty, and a
cross-encoder has no notion of citations — it will happily rank a
topically-nicer neighbour above the exact section asked for. Those results
carry a `pinned` flag out of `hybrid_search` and are passed through untouched,
which is why the citation metrics above are flat rather than damaged.

**How many candidates to rerank depends on the reranker.** The standard advice
is to rerank far more than you return — 30+, so the cross-encoder can rescue a
hit buried at rank 25. With the small MiniLM model that was backwards
(prose, n = 84):

| candidates | recall@5  | recall@10 | MRR@10    |
| ---------- | --------- | --------- | --------- |
| hybrid     | 0.798     | **0.929** | 0.611     |
| **10**     | **0.845** | **0.929** | **0.691** |
| 12         | **0.845** | 0.893     | 0.687     |
| 15         | 0.833     | 0.893     | 0.682     |
| 20         | 0.821     | 0.893     | 0.680     |
| 30         | 0.798     | 0.893     | 0.677     |

Every candidate past 10 is a section hybrid already judged worse, and this
cross-encoder promotes enough of them past true hits to push three prose
answers out of the top 10 — while never pulling a new one in. The deeper pool
is pure downside. So the reranker earns its keep by *reordering what hybrid
already returns*, not by reaching deeper, and `DEFAULT_CANDIDATES` is 10.

With candidates == k, recall@10 *cannot* move — reordering ten results keeps
the same ten — so those were ordering gains only.

The stronger `bge-reranker-base` changed the answer, which is why the
candidate count is a tuned parameter rather than a constant. Its re-sweep
(cross-validated, see [Parameters are re-tuned, honestly](#parameters-are-re-tuned-honestly))
settled on **12**, and that is where recall@10 finally moved: 0.908 → 0.949
on prose, four answers pulled up from ranks 11–12 that the small model could
not have been trusted with. The same sweep also switched on the
retriever-agreement gate (`agreement_scale` 0 → 0.5) that had been a wash
with MiniLM.

### Which cross-encoder

The first reranker was `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB, 6
layers). The obvious next experiment was a larger one, and because the
harness measures rerankers with no code changes it cost one flag:

```bash
uv run python -m eval.run --mode rerank --rerank-model BAAI/bge-reranker-base
```

Same 127 questions, same candidates, same parameters:

| prose (n = 98)      | MiniLM-L-6 | **bge-reranker-base** |
| ------------------- | ---------- | --------------------- |
| recall@5            | 0.837      | **0.867**             |
| recall@10           | 0.908      | 0.908                 |
| MRR@10              | 0.664      | **0.696**             |
| citation recall@5   | 0.966      | **1.000**             |
| latency per query\* | **1.6 s**  | 8.3 s                 |

\*end to end on CPU, warm.

The gain is about three questions' worth, beyond the 1/n noise margin — but it
is not uniform, and that is worth seeing. Per question, 17 improved, 19
worsened, 91 did not move. The improvements are rescues from deep in the list:

```
  7 -> 1   31 CFR § 315.90   Can the Treasury waive its own savings-bond rules to spare someone…
  7 -> 1   1 CFR § 15.3      If someone at an agency is unsure how to word a new rule, can they get guidance…
  6 -> 1   4 CFR § 6.1       What are the rules for vacation and sick time for GAO staff?
```

The regressions are one- or two-place slips (rank 1 → 3). A model that reads
more carefully finds answers the small one could not, and occasionally
second-guesses an easy one. On balance it wins, so it is now the default; the
small model stays one flag away for when latency matters more.

**The cost is real.** Five times slower per query on CPU is the difference
between an interactive tool and a batch one. Two things make it acceptable
today: there is no serving SLA yet (the answer layer will add seconds of LLM
time regardless), and on a GPU both models are well under 100 ms. If that
changes, the trade is one line.

Because `rerank_candidates` is a property of the reranker — a stronger model
may earn a wider pool — the reranker is recorded in `retrieval/params.json`'s
provenance, and changing it makes the parameters stale exactly as growing the
corpus does. The tune stage re-swept the candidate count with the new model and, for the
first time, adopted a change — cross-validated gain +0.010 against a 0.008
margin. The combined effect of model plus re-tune, on the same 143 questions:

| prose (n = 98) | MiniLM, 10 cands | bge, 12 cands + agreement gate |
| -------------- | ---------------- | ------------------------------ |
| recall@5       | 0.837            | **0.898**                      |
| recall@10      | 0.908            | **0.949**                      |
| MRR@10         | 0.664            | **0.718**                      |

MRR +0.054 is about five questions' worth. The regression gate confirmed every
other mode unchanged to three decimals, which is the check that the gain came
from the reranker and not from a change in what it was fed.

Scores from either model are raw logits — unbounded, and comparable only
within one query's candidate list, never across queries.

One caveat on the evidence: an earlier run of this same comparison against a
Title-4-only index (245 chunks, 54 answerable questions) showed the reranker
*losing* to hybrid on every metric. On a corpus that small hybrid is already
near ceiling and there is little left to reorder. The gains above are real but
they are a property of this corpus size; re-measure when it grows again.

## BM25 tokenization

BM25 originally split on `[a-z0-9]+`, which destroys exactly the strings a
regulation search has to get right:

```
"§ 2.3(b)(1)"  ->  ["2", "3", "b", "1"]        "4 CFR 25.14"  ->  ["4", "cfr", "25", "14"]
```

Those fragments are useless — "2" and "3" match nearly every section in the
corpus. `index/tokenizer.py` keeps citations whole and *normalizes* them, so
every way of writing the same reference collapses onto one token:

```
"§ 25.14"  "sec. 25.14"  "section 25.14"  "4 CFR 25.14"  "25.14"   ->  "§25.14"
```

A subdivided citation also emits its parent (`§ 2.3(b)(1)` → `§2.3(b)(1)` *and*
`§2.3`), so a query for a section matches text citing one of its paragraphs.
Statute cites stay in their own namespace: `29 U.S.C. 206` → `29usc206`, which
must not collide with `4 CFR § 206`.

**The tokenizer alone fixed nothing.** A section's body almost never contains
its own number — the text of `4 CFR § 25.14` opens "No person while entering…"
and never says "25.14" (7 of Title 4's 245 chunks cite themselves). BM25
indexed only the body, so a section was unfindable by its own citation no
matter how well citations tokenized. BM25 now indexes the citation and heading
path too — the same contextualized document the embedder already used, which
had made the two retrievers asymmetric.

Retrieving each of Title 4's 221 sections by citing it (hit@1):

| Query          | before | tokenizer only | tokenizer + full document |
| -------------- | ------ | -------------- | ------------------------- |
| `§ 25.14`      | 0.01   | 0.01           | **0.93**                  |
| `4 CFR 25.14`  | 0.01   | 0.01           | **1.00**                  |
| `sec. 25.14`   | 0.01   | 0.01           | **0.93**                  |

Hybrid search then diluted this back to ~0.6, because the vector retriever has
no notion of citations and returned neighbouring sections that outranked the
exact match. Naming a section is a known-item lookup rather than a similarity
search, so an exact citation match now sorts above ranked similarity — taking
hybrid to **1.00** hit@1.

A citation in the query also overrides the BM25 confidence gate described
under [Hybrid fusion](#hybrid-fusion). That gate infers whether BM25 found a
real lexical anchor from the shape of its scores, because normally there is
nothing better to go on; a citation answers the question directly. Without the
override, a question citing a section whose score curve happened to look flat
lost BM25 entirely — discarding the one ranking that knew where that section
was referenced.

The override applies only when the cited section is actually in the index —
or *referenced* by it: a section interior to a `§§ X through Y` range
(§ 1010.615, covered by § 1010.605's range but no section itself) exists in
the BM25 vocabulary only through range expansion, and the documents holding
that token are the only possible answers to the lookup, so they are promoted
just below true exact matches. A citation can instead name a section that
does not exist at all (`4 CFR 4.999`) or one from a title that was never
ingested; in both cases the citation tokens match nothing and BM25 is left
ranking the question's leftover prose, which is exactly the noise the gate
exists to catch. Forcing full trust
there made it worse: on the § 4.999 negative, BM25's natural confidence was
0.37, the unconditional override pushed § 4.3 (*Removal for unacceptable
performance*) into the top 5 on keyword overlap alone, and restricting the
override to indexed sections returns hybrid to semantic order. The gold set
carries both negatives, so a regression here shows up in their recorded
rankings. On the Title 31 negative the gate already rated BM25 fully
confident, so the override was moot — the gate's own false positives are a
separate problem.

All of these behaviours are inert for queries naming no section, so the prose
metrics above are untouched by them.

### Section ranges

Regulations delegate by range: § 28.131 says a petition "shall be processed in
accordance with §§ 28.19 through 28.25". Indexing only the endpoints leaves
every section in between unfindable, so a lookup for § 28.20 misses the text
that plainly governs it. Ranges now expand to one token per section:

```
"§§ 28.42 through 28.45"  ->  ["§28.42", "§28.43", "§28.44", "§28.45"]
```

Querying a mid-range section's citation and asking for the section that
governs it, with and without expansion — BM25 rank of the governing section:

| Query     | Governing section | endpoints only | expanded |
| --------- | ----------------- | -------------- | -------- |
| `§ 28.44` | § 28.22           | not found      | **3**    |
| `§ 28.20` | § 28.131          | not found      | **3**    |
| `§ 28.50` | § 28.11           | not found      | **3**    |
| `§ 28.70` | § 28.99           | not found      | **1**    |

Expansion is capped at **100 sections** (`RANGE_EXPANSION_CAP`). The widest
real range in Title 4 spans 73 numbers (`§§ 28.18 through 28.90`), so a tighter
cap would truncate legitimate ranges and reintroduce the bug for their tails.
The cap exists to bound *malformed* input: a garbled `§§ 1.1 through 1.99999`
would otherwise emit 100,000 tokens into a single document. At 100 the worst
case adds about half a median chunk (207 tokens), which BM25's document-length
normalization absorbs. A range over the cap falls back to its endpoints rather
than expanding partway — covering the start of a range and silently not the
rest is worse than not expanding at all. Ranges are only expanded when the
endpoints differ in their last component alone, so `§§ 28.18 through 31.5` is
left as two citations rather than inventing 300 sections.

### Stemming

Without stemming, "employees" in a question misses "employee" in the
regulation. The stemmer is deliberately plural-only (an S-stemmer): it folds
`employees → employee` and `copies → copy` and stops there.

Heavier stemming was measured, not assumed. On this gold set a full Porter2
stemmer scored slightly *better* (BM25 recall@5 0.64 vs 0.62, hybrid MRR 0.69
vs 0.69) — but that is a one-question difference across 42 questions, and
Porter2 conflates terms a regulation distinguishes:

| Stem        | Porter2 merges                                                    |
| ----------- | ----------------------------------------------------------------- |
| `applic`    | applicable, **applicant**, applicants, application, applicability  |
| `repres`    | represent, **representative**, represented, representing           |
| `administr` | administration, administrative, **administrator**                  |
| `time`      | time, times, **timely**, timing                                    |

An "applicant" is a person seeking a job; an "applicable" rule is not. Adding
`-ing`/`-ed` stripping instead merges `building → build`, and Part 25 is
entirely about the GAO *Building*. Those collisions cost precision in ways 42
questions cannot detect, so the conservative stemmer is the default; `tokenize`
takes a `stemmer=` argument for re-testing this on a larger gold set.

### Stopwords

English function words — articles, prepositions, pronouns, auxiliaries, and
the modals ("shall", "may", "must") that open nearly every regulatory
sentence — are dropped from documents and queries alike. This is not the
usual tidy-up: rank_bm25 clamps a negative-IDF term (one in more than half
the corpus) to `epsilon × average_idf` instead of zero, and that floor
*grows* with the corpus, because rarer vocabulary pushes the average IDF up.
On Title 4's 245 chunks the effect was noise; on the two-title corpus (6,262
chunks) a stopword-heavy section outscored a document holding a unique
citation token — "Where are the terms used in § 1010.615 defined?" ranked
the one section containing `§1010.615` at 71, beneath sections matching only
"where/the/term/used". Since rank_bm25 does not expose its epsilon, the fix
is to remove stopwords from the vocabulary entirely. Content words stay,
however common. On the gold set this lifted BM25's prose recall@5 from 0.51
to 0.61 with recall@10 and both citation metrics unharmed.

The same corpus growth also broke the confidence gate's clean calibration:
on Title 4 alone, real lexical matches showed contrast 1.9–3.8 and noise
1.2–1.5, but on 108 questions over two titles the two distributions overlap
across 1.2–1.9. No threshold pair is cleanly right anymore, so the ramp was
widened (1.4 → 2.2) to span the overlap — graded trust instead of a hard
call — picked by grid search on the gold set. The plateau is shallow;
re-tune with the next corpus or gold-set change.

### Re-indexing

Changing the analyzer invalidates a stored BM25 index — a query tokenized one
way cannot search terms built the other. `index/tokenizer.py` carries an
`ANALYZER_VERSION` that is stored alongside the index and checked on load, so
a stale index rebuilds itself rather than silently returning garbage. Bump it
whenever `tokenize` or `bm25_document` changes output, and re-tune the BM25
confidence thresholds in `retrieval/search.py` with it: they are calibrated
against the score distribution the analyzer produces.

## Development

```bash
uv run pytest                 # full test suite
uv run pytest -m "not slow"   # offline tests only (what CI runs)
uv run ruff check .
uv run python -m pipeline.manifest    # what the corpus declares vs what exists
uv run python -m eval.goldset --check # gold-set schema + coverage (needs no index)
uv run python -m retrieval.tune --dry-run   # would re-tuning change anything?
```

CI (GitHub Actions) runs ruff, the offline tests, and the manifest and
gold-set gates on every push. The corpus itself is not checked there — the
index is gitignored and rebuilding it would mean downloading several titles
per push — but the manifest and gold set are committed, and their gates need
no index. Run `python -m pipeline.onboard` locally to gate the corpus.

Raw eCFR downloads (`data/raw/`), chunks, indexes and eval outputs are
gitignored. Everything can be rebuilt with the commands above.

## Data

All regulation text comes from the public eCFR service run by the US Office of
the Federal Register. RegSearch does not handle any personal data. Its output is
not legal advice: always check the official text on
[ecfr.gov](https://www.ecfr.gov).

## License

[MIT](LICENSE)
