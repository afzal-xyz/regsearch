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
  a BM25 index in one SQLite database. BM25 uses a citation-aware analyzer
  (see [BM25 tokenization](#bm25-tokenization)).
- **`retrieval/`**: vector, BM25 and hybrid search. The hybrid mode fuses the
  two rankings over *sections*, weighting BM25 by how much it can be trusted on
  that particular query (see [Hybrid fusion](#hybrid-fusion)).
- **`eval/`**: a hand-written gold set of 53 questions (`eval/gold.jsonl`) —
  prose questions, citation lookups, and some with no answer in the corpus —
  plus a harness that reports recall@5, recall@10 and MRR@10 per retrieval
  mode, broken out by question `kind`.

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
`data/index/regsearch.db`. Re-running this is also how you pick up a change to
the BM25 analyzer; an index left over from an older one rebuilds itself on the
next query (see [Re-indexing](#re-indexing)).

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

The gold set holds two kinds of question, reported separately because they are
different tasks: 42 phrased in everyday language, and 6 known-item lookups that
name a section outright.

**Prose questions** (n = 42):

| Mode   | recall@5 | recall@10 | MRR@10   |
| ------ | -------- | --------- | -------- |
| vector | 0.81     | 0.90      | 0.66     |
| bm25   | 0.62     | 0.69      | 0.45     |
| hybrid | **0.86** | **0.93**  | **0.68** |

Hybrid beats vector-only on all three, and 0.93 is the ceiling set by the union
of what either retriever finds at all.

**Citation lookups** (n = 6):

| Mode   | recall@5 | recall@10 | MRR@10   |
| ------ | -------- | --------- | -------- |
| vector | 0.33     | 0.50      | 0.35     |
| bm25   | **1.00** | **1.00**  | **0.88** |
| hybrid | 0.83     | **1.00**  | 0.77     |

Vector search is near-useless here — embeddings have no notion of a citation —
which is why BM25 carries this case. See
[BM25 tokenization](#bm25-tokenization).

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
improves (0.81 → 0.83) and MRR holds at parity; raising MRR is the reranker's job.

Two caveats worth knowing: the gains hold across a wide parameter plateau
(`rrf_k` 5–60, pools 30–150, contrast thresholds 1.4–1.7), so they come from the
structure rather than tuned constants — but 42 answerable questions over one small
title is a thin basis for the thresholds, which should be re-tuned as the gold set
grows. And because vector dominates BM25 so heavily on this corpus, hybrid is
deliberately skewed semantic; a corpus with more exact-term queries would want the
BM25 weight raised.

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

Both behaviours are inert for queries naming no section, so the prose metrics
above are untouched by them.

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
