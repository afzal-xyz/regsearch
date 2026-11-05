"""Hybrid retrieval over the RegSearch index.

Three entry points, each returning ranked :class:`SearchResult`s (citation,
score, and which retrievers hit):

- :func:`vector_search` — semantic KNN via sqlite-vec.
- :func:`bm25_search`   — lexical BM25 via rank-bm25.
- :func:`hybrid_search` — both, fused with Reciprocal Rank Fusion.

Hybrid fusion has three properties that plain RRF over chunks does not. Each
was measured against ``eval/gold.jsonl``; see "Hybrid fusion" in the README.

1. **Fusion happens over sections, not chunks.** A section split into several
   chunks used to occupy several RRF slots and crowd out whole other sections.
   Each retriever's ranking is collapsed to its best chunk per section first,
   so a section's rank means the same thing on both sides of the fusion.

2. **BM25 is weighted by its confidence on this query** (see
   :func:`bm25_confidence`). BM25 scores every query against *something*, so on
   a question phrased in everyday words it returns a confident-looking but
   arbitrary ranking. Fusing that at full strength drags good semantic hits
   down. The weight scales with how sharply BM25's top hit stands out.

3. **A small ``rrf_k``.** The classic k = 60 flattens ranks 1..50 into a
   narrow score band, so the distinction between "best hit" and "50th hit"
   nearly vanishes and co-occurrence in two pools outweighs it.

CLI:
    python -m retrieval.search "when must a GAO employee file a financial disclosure" --k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass

import numpy as np
import sqlite_vec

from index import store
from index.embedder import DEFAULT_MODEL, Embedder
from index.store import DEFAULT_DB, connect, load_bm25, tokenize
from index.tokenizer import citation_tokens
from retrieval import params as fusion_params
from retrieval.params import FusionParams

logger = logging.getLogger(__name__)

# The constants below are the hand-tuned DEFAULTS. The live values come from
# retrieval/params.json via retrieval.params (see that module), which
# retrieval.tune rewrites against the current corpus and gold set.
RRF_K = 10  # reciprocal rank fusion constant

# Candidate sections pulled from each retriever before fusion.
POOL = 50
# Chunks fetched per candidate section, so `POOL` distinct sections survive de-duping.
_POOL_OVERFETCH = 3

BM25_WEIGHT = 1.0  # weight of a fully-confident BM25 ranking (vector is always 1.0)

# BM25 confidence is measured as "contrast": the top score over the score at
# rank BM25_CONTRAST_ANCHOR. Below _LO the ranking is treated as noise and
# ignored; above _HI it is trusted fully; in between the weight ramps linearly.
#
# These are calibrated against the analyzer in index.tokenizer: changing how
# text is tokenized moves the score distribution, so re-tune them (and bump
# ANALYZER_VERSION) alongside any analyzer change.
#
# Retuned for analyzer v5 (stopword removal) on the two-title (4 + 31) index.
# The clean separation measured on Title 4 alone — real matches 1.9-3.8,
# noise 1.2-1.5 — does not survive a 6,262-chunk corpus: on the 108-question
# gold set the two distributions now overlap across 1.2-1.9, so no threshold
# pair is cleanly right and the ramp is widened to span the overlap, trading
# hard decisions for graded trust. Chosen by grid search on hybrid metrics
# (eval/gold.jsonl); the plateau is shallow, so expect to re-tune again when
# the corpus or gold set grows.
BM25_CONTRAST_ANCHOR = 20
BM25_CONTRAST_LO = 1.4
BM25_CONTRAST_HI = 2.2


@dataclass
class SearchResult:
    chunk_id: str
    citation: str  # chunk citation, may include a paragraph anchor
    section_citation: str  # section-level citation (the eval hit key)
    score: float
    retrievers: list[str]  # subset of {"vector", "bm25"} that surfaced this chunk
    heading: str
    heading_path: list[str]
    text: str  # pure chunk text
    # Set by :func:`hybrid_search` when this section's rank is a certainty
    # rather than a similarity: the query named it outright, or its text
    # references the cited section. Downstream re-ranking must leave these
    # alone — see :mod:`rag.reranker`. Always False from the single-retriever
    # entry points, which have no notion of citations.
    pinned: bool = False


def _load_chunk(conn: sqlite3.Connection, chunk_id: str) -> sqlite3.Row:
    return conn.execute("select * from chunks where chunk_id = ?", (chunk_id,)).fetchone()


def _result(
    conn: sqlite3.Connection,
    chunk_id: str,
    score: float,
    retrievers: list[str],
    pinned: bool = False,
) -> SearchResult:
    row = _load_chunk(conn, chunk_id)
    return SearchResult(
        chunk_id=chunk_id,
        citation=row["citation"],
        section_citation=row["section_citation"],
        score=float(score),
        retrievers=retrievers,
        heading=row["heading"],
        heading_path=json.loads(row["heading_path"]),
        text=row["text"],
        pinned=pinned,
    )


def vector_search(
    query: str, k: int = 5, *, conn: sqlite3.Connection, embedder: Embedder
) -> list[SearchResult]:
    store.assert_model_matches(conn, embedder)
    qv = embedder.encode_query(query)
    knn = conn.execute(
        "select rowid, distance from vec_chunks where embedding match ? order by distance limit ?",
        (sqlite_vec.serialize_float32(np.asarray(qv, dtype=np.float32).tolist()), k),
    ).fetchall()
    if not knn:
        return []
    rowids = [r["rowid"] for r in knn]
    placeholders = ",".join("?" * len(rowids))
    chunk_rows = conn.execute(
        f"select id, chunk_id from chunks where id in ({placeholders})", rowids
    )
    id_to_chunk = {r["id"]: r["chunk_id"] for r in chunk_rows}
    results = []
    for r in knn:
        chunk_id = id_to_chunk.get(r["rowid"])
        if chunk_id is None:
            continue
        # normalized embeddings: cosine similarity = 1 - L2^2 / 2
        similarity = 1.0 - (r["distance"] ** 2) / 2.0
        results.append(_result(conn, chunk_id, similarity, ["vector"]))
    return results


def bm25_search(query: str, k: int = 5, *, conn: sqlite3.Connection) -> list[SearchResult]:
    ids, model = load_bm25(conn)
    if model is None or not ids:
        return []
    scores = model.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:k]
    return [_result(conn, ids[i], scores[i], ["bm25"]) for i in order if scores[i] > 0]


def bm25_confidence(
    scores: list[float],
    *,
    anchor: int = BM25_CONTRAST_ANCHOR,
    lo: float = BM25_CONTRAST_LO,
    hi: float = BM25_CONTRAST_HI,
) -> float:
    """How much to trust a BM25 ranking, in [0, 1], from its score distribution.

    BM25 ranks every query against *something*: ask "can I bring a gun into the
    GAO building?" and it will happily return 50 chunks, none of which contain
    "gun" (the regulation says "firearms"). The tell is not the absolute score
    but the *contrast* — a real lexical match towers over the rest of the list,
    while noise comes back nearly flat.

    Contrast is the top score divided by the score at rank ``anchor``, which is
    scale-free: it does not move with query length or corpus size. It ramps
    linearly from 0 at ``lo`` to 1 at ``hi``. The ramp matters — a hard
    threshold scores the same recall but a visibly worse MRR, because queries
    land either side of the cutoff by a hair.
    """
    if not scores or scores[0] <= 0:
        return 0.0
    if len(scores) < anchor:
        # Fewer than `anchor` chunks share any term with the query at all, so
        # the query is highly discriminative — exactly when BM25 is worth most.
        return 1.0
    reference = scores[anchor - 1]
    if reference <= 0:
        return 1.0
    contrast = scores[0] / reference
    if contrast <= lo:
        return 0.0
    if contrast >= hi:
        return 1.0
    return (contrast - lo) / (hi - lo)


# Per-connection cache for :func:`indexed_section_tokens`, keyed by ``id(conn)``
# and invalidated when the chunk count changes.
_SECTION_TOKEN_CACHE: dict[int, tuple[int, frozenset[str]]] = {}


def indexed_section_tokens(conn: sqlite3.Connection) -> frozenset[str]:
    """Citation tokens of every section in the index (``{"§25.14", "4cfr25.14", ...}``).

    Lets a caller ask "is the section this query names actually in the
    corpus?", which :func:`citation_tokens` alone cannot answer — it will
    happily extract ``§4.999`` from a question about a section that does not
    exist. Cached per connection; the cache is dropped when the chunk count
    changes, so a rebuilt index is picked up without reconnecting.
    """
    n = conn.execute("select count(*) from chunks").fetchone()[0]
    cached = _SECTION_TOKEN_CACHE.get(id(conn))
    if cached and cached[0] == n:
        return cached[1]
    rows = conn.execute("select distinct section_citation from chunks").fetchall()
    tokens = frozenset().union(*(citation_tokens(r[0]) for r in rows)) if rows else frozenset()
    _SECTION_TOKEN_CACHE[id(conn)] = (n, tokens)
    return tokens


def _citing_chunks(conn: sqlite3.Connection, tokens: frozenset[str]) -> list[tuple[str, str]]:
    """Chunks whose BM25 document contains any of ``tokens``, best match first.

    A citation token can be in the BM25 vocabulary without naming any indexed
    section: range expansion writes §1010.615 into § 1010.605's document
    because its text covers "§§ 1010.610 through 1010.630". Such a token is
    still a verified lexical anchor — some indexed text references exactly the
    section the query asks about — and this returns that text, as
    ``(chunk_id, section_citation)`` pairs scored on the citation tokens
    alone: the question's prose must not dilute what is a pure "which text
    references this section" lookup. Empty when no token is in the
    vocabulary (a truly non-existent citation, § 4.999).
    """
    ids, model = load_bm25(conn)
    if model is None or not ids:
        return []
    scores = model.get_scores(sorted(tokens))
    order = np.argsort(scores)[::-1]
    out = []
    for i in order:
        if scores[i] <= 0:
            break
        row = conn.execute(
            "select section_citation from chunks where chunk_id = ?", (ids[i],)
        ).fetchone()
        if row:
            out.append((ids[i], row["section_citation"]))
    return out


def agreement(
    vector_hits: list[SearchResult], bm25_hits: list[SearchResult], k: int, scale: float
) -> float:
    """Fraction in [0, 1]: top-``k`` section overlap over ``scale``, capped at 1.

    ``scale`` is the overlap (as a fraction of k) at which BM25 is fully
    trusted: with k=10 and scale=0.3, sharing 3+ sections means full trust,
    1 section a third, none zero.
    """
    if k <= 0 or scale <= 0:
        return 1.0
    a = {r.section_citation for r in vector_hits[:k]}
    b = {r.section_citation for r in bm25_hits[:k]}
    return min(1.0, (len(a & b) / k) / scale)


def _best_per_section(results: list[SearchResult]) -> list[SearchResult]:
    """Collapse a ranked chunk list to its best chunk per section, order kept.

    The list is already best-first, so the first chunk seen for a section is
    that section's best match.
    """
    seen: set[str] = set()
    collapsed = []
    for r in results:
        if r.section_citation in seen:
            continue
        seen.add(r.section_citation)
        collapsed.append(r)
    return collapsed


@dataclass
class Candidates:
    """Everything :func:`fuse` needs, fetched once per query.

    Retrieval (embedding the query, the sqlite-vec KNN, BM25 over every chunk,
    the citation lookups) is the expensive half of hybrid search and depends on
    nothing tunable. Fusion is arithmetic over these lists. Splitting them lets
    :mod:`retrieval.tune` fetch candidates for every gold question once and
    then score hundreds of parameter sets in seconds, instead of re-running
    retrieval per parameter set.
    """

    query: str
    vector_hits: list[SearchResult]  # best chunk per section, vector order
    bm25_chunks: list[SearchResult]  # raw chunk ranking (confidence reads its scores)
    known: frozenset[str]  # citation tokens in the query that name indexed sections
    citing: list[tuple[str, str]]  # (chunk_id, section) referencing cited-but-absent sections


def retrieve_candidates(
    query: str,
    *,
    conn: sqlite3.Connection,
    embedder: Embedder,
    pool: int = POOL,
) -> Candidates:
    """Run both retrievers and the citation lookups; no fusion yet."""
    chunk_pool = pool * _POOL_OVERFETCH
    vector_hits = _best_per_section(vector_search(query, chunk_pool, conn=conn, embedder=embedder))
    bm25_chunks = bm25_search(query, chunk_pool, conn=conn)

    # Sections the query names that are actually in the index. A query can
    # cite a section that does not exist (§ 4.999) or lives in a title that
    # was never ingested; those are *not* a lexical anchor.
    wanted = citation_tokens(query)
    known = wanted & indexed_section_tokens(conn) if wanted else frozenset()

    # A cited section can be absent from the index yet *referenced* by indexed
    # text — most often a section interior to a "§§ X through Y" range, whose
    # token exists only because the analyzer expanded the range. The documents
    # containing that token are the only possible answers to the lookup, so
    # they are promoted below true exact matches in `fuse`. A token in neither
    # the sections nor the vocabulary (§ 4.999) still gets no special treatment.
    citing = _citing_chunks(conn, wanted - known) if wanted - known else []
    return Candidates(query, vector_hits, bm25_chunks, known, citing)


def fuse(
    cands: Candidates,
    k: int,
    *,
    params: FusionParams,
    pool: int = POOL,
    bm25_weight: float | None = None,
) -> list[tuple[str, str, float, list[str], bool]]:
    """Confidence-weighted, section-level RRF over ``cands``.

    Returns ``(section, chunk_id, score, retrievers, pinned)`` tuples, best
    first, without touching the database — so the tuner can call it in a tight
    loop. :func:`hybrid_search` turns them into :class:`SearchResult`s.

    Each retriever contributes ``weight / (rrf_k + rank)`` per section, with
    1-based ranks over its ``pool`` best *sections*. Vector always has weight
    1.0; BM25 has ``bm25_weight`` scaled by :func:`bm25_confidence`, so a BM25
    ranking that looks like noise is damped or ignored and hybrid falls back
    to pure semantic order — unless the query names a section that is in the
    index (or referenced by indexed text), in which case BM25 has a verified
    lexical anchor and is trusted outright.
    """
    weight = params.bm25_weight if bm25_weight is None else bm25_weight
    if cands.known or cands.citing:
        # bm25_confidence infers "did BM25 find a real lexical anchor" from the
        # shape of its scores, because normally there is nothing better to go
        # on. A query naming an indexed section answers that directly, and a
        # citation is the most discriminating term there is — do not let the
        # heuristic veto it. The check against the index matters: when the
        # cited section is absent, BM25 is ranking the question's leftover
        # prose, which is exactly the noise the gate exists to catch.
        confidence = 1.0
    else:
        confidence = bm25_confidence(
            [r.score for r in cands.bm25_chunks],
            anchor=params.bm25_contrast_anchor,
            lo=params.bm25_contrast_lo,
            hi=params.bm25_contrast_hi,
        )
    bm25_hits = _best_per_section(cands.bm25_chunks)
    if confidence > 0 and params.agreement_scale > 0 and not (cands.known or cands.citing):
        # Score contrast says how *sure* BM25 is; it cannot say whether BM25 is
        # sure about the right thing. On Title 1's prose BM25 was confidently
        # wrong — its top 10 held the answer for 1 question in 9, yet it was
        # trusted at ~40% and nudged vector's correct rank-1 down to 2 or 3.
        # Whether the two retrievers broadly agree is an independent signal:
        # when BM25 is actually useful its top 10 shares several sections with
        # vector's; when it is noise, almost none.
        confidence *= agreement(cands.vector_hits, bm25_hits, params.agreement_k,
                                params.agreement_scale)
    bm25_scaled = weight * confidence

    fused: dict[str, float] = {}
    best_chunk: dict[str, str] = {}
    retrievers: dict[str, set[str]] = {}
    for ranked, w in ((cands.vector_hits, 1.0), (bm25_hits, bm25_scaled)):
        if w <= 0:
            continue
        for rank, res in enumerate(ranked[:pool], start=1):
            section = res.section_citation
            fused[section] = fused.get(section, 0.0) + w / (params.rrf_k + rank)
            best_chunk.setdefault(section, res.chunk_id)
            retrievers.setdefault(section, set()).update(res.retrievers)

    # The cited-but-absent case: inject the sections whose text references the
    # cited number, in citation-match order, even when neither pool surfaced
    # them — at 6k+ chunks a single citation occurrence in a long document
    # scores below tf-stacked common words.
    promoted: dict[str, int] = {}
    for rank, (chunk_id, section) in enumerate(cands.citing, start=1):
        if section not in promoted:
            promoted[section] = rank
            fused.setdefault(section, 0.0)
            best_chunk.setdefault(section, chunk_id)
            retrievers.setdefault(section, set()).add("bm25")

    # A query naming a section is a known-item lookup, not a search: an exact
    # citation match is certainty, and ranked similarity cannot outvote it.
    # Sections merely *referencing* a cited-but-absent number rank just below.
    known = cands.known
    exact = {s for s in fused if known & citation_tokens(s)} if known else frozenset()

    top = sorted(
        fused.items(),
        key=lambda kv: (kv[0] in exact, kv[0] in promoted, -promoted.get(kv[0], 0), kv[1]),
        reverse=True,
    )[:k]
    return [
        (
            section,
            best_chunk[section],
            score,
            sorted(retrievers[section]),
            section in exact or section in promoted,
        )
        for section, score in top
    ]


def hybrid_search(
    query: str,
    k: int = 5,
    *,
    conn: sqlite3.Connection,
    embedder: Embedder,
    params: FusionParams | None = None,
    pool: int = POOL,
    bm25_weight: float | None = None,
    rrf_k: int | None = None,
) -> list[SearchResult]:
    """Hybrid retrieval: :func:`retrieve_candidates` then :func:`fuse`.

    ``params`` defaults to the tuned set in ``retrieval/params.json`` (or the
    code defaults when that file is absent). ``bm25_weight`` / ``rrf_k`` are
    per-call overrides kept for experiments and tests.

    Returns up to ``k`` results, one per section — the section's best-matching
    chunk, carrying the fused score.
    """
    params = params or fusion_params.current()
    if rrf_k is not None:
        params = params.with_(rrf_k=rrf_k)
    cands = retrieve_candidates(query, conn=conn, embedder=embedder, pool=pool)
    fused = fuse(cands, k, params=params, pool=pool, bm25_weight=bm25_weight)
    return [
        _result(conn, chunk_id, score, retrievers, pinned=pinned)
        for _section, chunk_id, score, retrievers, pinned in fused
    ]


def _snippet(text: str, n: int = 200) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("…" if len(flat) > n else "")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Citations contain "§"; make sure they render on a non-UTF-8 Windows console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="python -m retrieval.search",
        description="Hybrid (vector + BM25) search over the RegSearch index.",
    )
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    # A fetch/model download uses HTTPS; honor the OS trust store locally.
    from ingest import fetcher

    fetcher._use_os_trust_store()

    conn = connect(args.db)
    try:
        embedder = Embedder(args.model)
        results = hybrid_search(args.query, args.k, conn=conn, embedder=embedder)
    except store.ModelMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if not results:
        print("(no results — is the index built?)")
        return 0

    for i, r in enumerate(results, start=1):
        hits = "+".join(r.retrievers)
        print(f"{i}. {r.citation}   [{hits}]  score={r.score:.4f}")
        print(f"   {_snippet(r.text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
