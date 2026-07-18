"""Hybrid retrieval over the RegSearch index.

Three entry points, each returning ranked :class:`SearchResult`s (citation,
score, and which retrievers hit):

- :func:`vector_search` — semantic KNN via sqlite-vec.
- :func:`bm25_search`   — lexical BM25 via rank-bm25.
- :func:`hybrid_search` — both, fused with Reciprocal Rank Fusion (k = 60).

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

logger = logging.getLogger(__name__)

RRF_K = 60  # reciprocal rank fusion constant


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


def hybrid_search(
    query: str,
    k: int = 5,
    *,
    conn: sqlite3.Connection,
    embedder: Embedder,
    rrf_k: int = RRF_K,
) -> list[SearchResult]:
    """Fuse vector + BM25 rankings with Reciprocal Rank Fusion.

    RRF score for a chunk = sum over retrievers of 1 / (rrf_k + rank), with
    1-based ranks. Pulls a pool of ``max(k, 50)`` from each retriever first.
    """
    pool = max(k, 50)
    vector_hits = vector_search(query, pool, conn=conn, embedder=embedder)
    bm25_hits = bm25_search(query, pool, conn=conn)

    fused: dict[str, float] = {}
    retrievers: dict[str, set[str]] = {}
    for ranked in (vector_hits, bm25_hits):
        for rank, res in enumerate(ranked, start=1):
            fused[res.chunk_id] = fused.get(res.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            retrievers.setdefault(res.chunk_id, set()).update(res.retrievers)

    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [_result(conn, cid, score, sorted(retrievers[cid])) for cid, score in top]


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
