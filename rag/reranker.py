"""Cross-encoder reranking of hybrid-retrieval candidates.

Hybrid retrieval (see :mod:`retrieval.search`) scores a query and a passage
*independently* — the query becomes one vector, every chunk became a vector at
index time, and the ranking is their similarity. That is what makes it fast
enough to search a whole corpus, and it is also its ceiling: the model never
sees the question and the passage together, so it cannot notice that a passage
mentions all the right words while answering a different question.

A cross-encoder does see them together. It takes the pair ``(query, passage)``
as a single input and returns one relevance score. It is far too slow to run
over a corpus — one forward pass per passage — but over the ~30 candidates
hybrid already narrowed to, it is cheap, and it reorders them with a much
better notion of "does this actually answer the question".

This module is the reorder step only; recall is whatever hybrid handed over.
If the right section is not in the candidate list, reranking cannot invent it
— and on this corpus, handing it a *wider* list than the user asked for makes
things worse rather than better (see :data:`DEFAULT_CANDIDATES`).

Two properties are deliberate:

1. **Pinned results are never reordered.** A query naming a section
   ("§ 25.14") is a known-item lookup, and hybrid already resolved it with
   certainty. A cross-encoder has no notion of citations and would happily
   rank a topically-nicer neighbour above the section the user asked for, so
   :attr:`~retrieval.search.SearchResult.pinned` results keep their order and
   stay on top. Measured on the citation questions in ``eval/gold.jsonl``,
   reranking them costs MRR outright.

2. **The passage scored is the contextualized chunk** — heading path, newline,
   text — the same form that was embedded. A bare chunk body often reads as a
   fragment ("(a) The term means ...") with no clue which section it is from;
   the heading path is what makes it legible to the cross-encoder.

CLI:
    python -m rag.reranker "when must a GAO employee file a financial disclosure" --k 5
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import replace

import numpy as np

from index import store
from index.embedder import DEFAULT_MODEL, Embedder, contextualize
from index.store import DEFAULT_DB, connect
from retrieval.search import SearchResult, _snippet, hybrid_search

logger = logging.getLogger(__name__)

# bge-reranker-base: 12 layers, ~1.1 GB. Measured against ms-marco-MiniLM-L-6-v2
# on eval/gold.jsonl (127 in-corpus questions, identical candidates and
# parameters): prose MRR@10 0.664 -> 0.696, prose recall@5 0.837 -> 0.867,
# citation recall@5 0.966 -> 1.000. Not uniform — 17 questions improved, 19
# worsened — but the gains are rescues from deep in the list (rank 7 -> 1)
# and the losses are one- or two-place slips, so the net is ~3 questions'
# worth, beyond the 1/n noise margin. The cost is latency: ~8.3 s per query
# against ~1.6 s for MiniLM, end to end on CPU. Scores are raw logits —
# unbounded, comparable only within one query's candidate list.
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
# The previous default, one flag away (`--rerank-model`) when latency matters
# more than the ~3-question gain.
SMALL_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Candidates pulled from hybrid before reranking — the CODE DEFAULT only. The
# live value is `rerank_candidates` in retrieval/params.json, which the tune
# stage re-sweeps whenever the corpus, gold set or reranker changes; with
# bge-reranker-base it settled on 12. The table below is the original sweep
# with ms-marco-MiniLM-L-6-v2 and is kept because its conclusion still holds
# for that model: a weak reranker is hurt by a wide pool.
#
# The textbook setting is "much wider than k" — rerank 30+ so the cross-encoder
# can rescue a hit hybrid buried at rank 25. Measured on eval/gold.jsonl
# (6,262 chunks, 108 in-corpus questions, k = 10) that is simply wrong here:
#
#   candidates   prose r@5   prose r@10   prose MRR@10
#   hybrid           0.798        0.929          0.611
#   10               0.845        0.929          0.691
#   12               0.845        0.893          0.687
#   15               0.833        0.893          0.682
#   20               0.821        0.893          0.680
#   30               0.798        0.893          0.677
#
# Widening the pool past 10 costs recall and gains nothing: every extra
# candidate is a section hybrid already judged worse, and this cross-encoder
# promotes enough of them past true hits to lose three prose questions out of
# the top 10 while never pulling a new one in. So the reranker earns its keep
# purely by *reordering what hybrid already returns*, not by reaching deeper.
#
# A consequence worth stating plainly: with candidates == k, recall@k cannot
# move — reordering ten results keeps the same ten. The gains above are real
# but they are ordering gains (MRR, and recall@5 within the top 10). Re-sweep
# this if the serving k changes, since the two are deliberately coupled.
DEFAULT_CANDIDATES = 10

# Pairs per cross-encoder forward batch.
DEFAULT_BATCH_SIZE = 32


class Reranker:
    """Lazy wrapper around a sentence-transformers ``CrossEncoder``.

    Mirrors :class:`index.embedder.Embedder`: the heavy import and the model
    load happen on first use, so importing this module (and tests that inject
    a fake) never pay the torch cost.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.model_name = model_name
        self._device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading cross-encoder %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self._device)
        return self._model

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance score per passage for ``query``. Higher is more relevant."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def rerank(
    query: str,
    results: list[SearchResult],
    k: int = 5,
    *,
    reranker: Reranker,
) -> list[SearchResult]:
    """Reorder ``results`` by cross-encoder relevance, best first, top ``k``.

    Pinned results (exact citation matches and range promotions) keep their
    incoming order ahead of everything else; only the rest is rescored. The
    returned results carry the cross-encoder score in
    :attr:`~retrieval.search.SearchResult.score`, replacing the fused RRF
    score — the two are on different scales and must not be mixed.
    """
    pinned = [r for r in results if r.pinned]
    rest = [r for r in results if not r.pinned]
    if not rest:
        return pinned[:k]

    scores = reranker.score(query, [contextualize(r) for r in rest])
    # argsort is ascending and stable, so sorting the negated scores keeps
    # hybrid's order as the tie-break among equally-scored candidates.
    order = np.argsort(-scores, kind="stable")
    reranked = [replace(rest[i], score=float(scores[i])) for i in order]
    return (pinned + reranked)[:k]


def rerank_search(
    query: str,
    k: int = 5,
    *,
    conn: sqlite3.Connection,
    embedder: Embedder,
    reranker: Reranker,
    candidates: int | None = None,
) -> list[SearchResult]:
    """Hybrid retrieval over ``candidates`` sections, reranked down to ``k``.

    ``candidates`` defaults to the tuned value in ``retrieval/params.json``
    (``rerank_candidates``; see :mod:`retrieval.tune`), and is raised to at
    least ``k`` so asking for more results than the pool still returns ``k``.
    """
    if candidates is None:
        from retrieval import params as fusion_params

        candidates = fusion_params.current().rerank_candidates
    pool = max(candidates, k)
    hits = hybrid_search(query, pool, conn=conn, embedder=embedder)
    return rerank(query, hits, k, reranker=reranker)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="python -m rag.reranker",
        description="Hybrid search with cross-encoder reranking.",
    )
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--candidates", type=int, default=None,
                        help="override the tuned rerank_candidates")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    args = parser.parse_args(argv)

    from ingest import fetcher

    fetcher._use_os_trust_store()

    conn = connect(args.db)
    try:
        results = rerank_search(
            args.query,
            args.k,
            conn=conn,
            embedder=Embedder(args.model),
            reranker=Reranker(args.rerank_model),
            candidates=args.candidates,
        )
    except store.ModelMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if not results:
        print("(no results — is the index built?)")
        return 0

    for i, r in enumerate(results, start=1):
        tag = "pinned" if r.pinned else f"score={r.score:.4f}"
        print(f"{i}. {r.citation}   [{'+'.join(r.retrievers)}]  {tag}")
        print(f"   {_snippet(r.text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
