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

logger = logging.getLogger(__name__)

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
BM25_CONTRAST_ANCHOR = 20
BM25_CONTRAST_LO = 1.5
BM25_CONTRAST_HI = 1.9


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


def _load_chunk(conn: sqlite3.Connection, chunk_id: str) -> sqlite3.Row:
    return conn.execute("select * from chunks where chunk_id = ?", (chunk_id,)).fetchone()


def _result(
    conn: sqlite3.Connection, chunk_id: str, score: float, retrievers: list[str]
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


def hybrid_search(
    query: str,
    k: int = 5,
    *,
    conn: sqlite3.Connection,
    embedder: Embedder,
    rrf_k: int = RRF_K,
    pool: int = POOL,
    bm25_weight: float = BM25_WEIGHT,
) -> list[SearchResult]:
    """Fuse vector + BM25 rankings with confidence-weighted, section-level RRF.

    Each retriever contributes ``weight / (rrf_k + rank)`` per section, with
    1-based ranks over its ``pool`` best *sections*. Vector always has weight
    1.0; BM25 has ``bm25_weight`` scaled by :func:`bm25_confidence`, so a BM25
    ranking that looks like noise is damped or ignored entirely and hybrid
    falls back to pure semantic order — unless the query names a section that
    is in the index, in which case BM25 has a verified lexical anchor and is
    trusted outright. A citation to a section the index does not hold (a
    non-existent number, or another title) gets no such treatment.

    Returns up to ``k`` results, one per section — the section's best-matching
    chunk, carrying the fused score.
    """
    chunk_pool = pool * _POOL_OVERFETCH
    vector_hits = _best_per_section(vector_search(query, chunk_pool, conn=conn, embedder=embedder))
    bm25_chunks = bm25_search(query, chunk_pool, conn=conn)

    # Sections the query names that are actually in the index. A query can
    # cite a section that does not exist (§ 4.999) or lives in a title that
    # was never ingested (31 CFR 1010.230); those are *not* a lexical anchor.
    wanted = citation_tokens(query)
    known = wanted & indexed_section_tokens(conn) if wanted else frozenset()
    if known:
        # bm25_confidence infers "did BM25 find a real lexical anchor" from the
        # shape of its scores, because normally there is nothing better to go
        # on. A query that names an indexed section gives us that answer
        # directly, and a citation is the most discriminating term there is —
        # so do not let the heuristic veto it. Without this, a question citing
        # a section whose score curve happens to look flat loses BM25
        # entirely, which is exactly the ranking that knows where that section
        # is referenced.
        #
        # The check against the index matters: when the cited section is
        # absent, the citation tokens match nothing and BM25 is ranking on the
        # question's remaining prose alone. Forcing full trust there injects
        # an arbitrary keyword ranking at full weight — measured on the
        # negative citation questions in eval/gold.jsonl, it pulled unrelated
        # sections into the top 5 that the confidence gate would have damped.
        confidence = 1.0
    else:
        # Read off the raw chunk scores, before section de-duping.
        confidence = bm25_confidence([r.score for r in bm25_chunks])
    bm25_scaled = bm25_weight * confidence
    bm25_hits = _best_per_section(bm25_chunks)

    fused: dict[str, float] = {}
    best_chunk: dict[str, str] = {}
    retrievers: dict[str, set[str]] = {}
    for ranked, weight in ((vector_hits, 1.0), (bm25_hits, bm25_scaled)):
        if weight <= 0:
            continue
        for rank, res in enumerate(ranked[:pool], start=1):
            section = res.section_citation
            fused[section] = fused.get(section, 0.0) + weight / (rrf_k + rank)
            best_chunk.setdefault(section, res.chunk_id)
            retrievers.setdefault(section, set()).update(res.retrievers)

    # A query that names a section ("§ 25.14", "4 CFR 25.14") is a known-item
    # lookup, not a search: an exact citation match is certainty, and ranked
    # similarity cannot outvote it. Without this the vector side — which has no
    # notion of citations and returns neighbouring sections — outranks the
    # requested section about half the time. No-op for queries naming no
    # section, or naming one that is not in the index.
    exact = {s for s in fused if known & citation_tokens(s)} if known else frozenset()

    top = sorted(fused.items(), key=lambda kv: (kv[0] in exact, kv[1]), reverse=True)[:k]
    return [
        _result(conn, best_chunk[section], score, sorted(retrievers[section]))
        for section, score in top
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
