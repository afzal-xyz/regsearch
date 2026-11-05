"""Tests for rag.reranker against the tiny fixture index, with a fake cross-encoder.

The fake scores a pair by keyword overlap, which is enough to assert the
*plumbing*: which passages get scored, in what form, what order comes back,
and that pinned results are left alone. Nothing here measures ranking quality
— that is eval/gold.jsonl's job.
"""

from __future__ import annotations

import pytest

from index import store
from rag import reranker as rr
from retrieval import search
from tests.test_index import FIXTURE_CHUNKS, FakeEmbedder

DISCLOSURE = "4 CFR § 7.3"
TRAVEL = "4 CFR § 8.1"


class FakeReranker:
    """Scores a pair by how many query words appear in the passage.

    Records every ``(query, passages)`` call so tests can assert what the
    cross-encoder was actually shown.
    """

    def __init__(self, boost: dict[str, float] | None = None):
        self.calls: list[tuple[str, list[str]]] = []
        self.boost = boost or {}  # substring -> score added when present

    def score(self, query, passages):
        import numpy as np

        self.calls.append((query, list(passages)))
        terms = set(query.lower().split())
        scores = []
        for p in passages:
            low = p.lower()
            s = float(sum(1 for t in terms if t in low))
            s += sum(v for key, v in self.boost.items() if key.lower() in low)
            scores.append(s)
        return np.asarray(scores, dtype=np.float32)


@pytest.fixture
def index(tmp_path):
    db_path = tmp_path / "index" / "regsearch.db"
    emb = FakeEmbedder()
    store.build_index(FIXTURE_CHUNKS, db_path=db_path, embedder=emb, show_progress=False)
    conn = store.connect(db_path)
    yield conn, emb
    conn.close()


def _hits(conn, emb, query, k=10):
    return search.hybrid_search(query, k, conn=conn, embedder=emb)


# --- rerank() ordering -----------------------------------------------------


def test_rerank_promotes_the_passage_the_cross_encoder_likes(index):
    conn, emb = index
    hits = _hits(conn, emb, "travel reimbursement expense")
    assert len(hits) > 1
    # Force the *last* hybrid result to the top; only the reranker can do this.
    underdog = hits[-1]
    reranker = FakeReranker(boost={underdog.text[:40]: 100.0})

    out = rr.rerank("travel reimbursement expense", hits, k=3, reranker=reranker)

    assert out[0].section_citation == underdog.section_citation
    assert out[0].score == pytest.approx(100.0, abs=10.0)


def test_rerank_returns_descending_scores_and_respects_k(index):
    conn, emb = index
    hits = _hits(conn, emb, "financial disclosure report")
    out = rr.rerank("financial disclosure report", hits, k=2, reranker=FakeReranker())
    assert len(out) == 2
    assert out[0].score >= out[1].score


def test_rerank_ties_keep_hybrid_order(index):
    conn, emb = index
    hits = _hits(conn, emb, "financial disclosure")

    class Flat(FakeReranker):
        def score(self, query, passages):
            import numpy as np

            return np.zeros(len(passages), dtype=np.float32)

    out = rr.rerank("financial disclosure", hits, k=len(hits), reranker=Flat())
    assert [r.section_citation for r in out] == [h.section_citation for h in hits]


def test_rerank_scores_the_contextualized_chunk(index):
    """The cross-encoder must see the heading path, not a bare chunk body."""
    conn, emb = index
    hits = _hits(conn, emb, "financial disclosure")
    reranker = FakeReranker()
    rr.rerank("financial disclosure", hits, k=5, reranker=reranker)

    _query, passages = reranker.calls[0]
    assert len(passages) == len(hits)
    for passage, hit in zip(passages, hits, strict=True):
        assert passage.startswith(" > ".join(hit.heading_path))
        assert hit.text in passage


def test_rerank_handles_empty_candidates():
    assert rr.rerank("anything", [], k=5, reranker=FakeReranker()) == []


# --- pinned results --------------------------------------------------------


def test_pinned_citation_hit_survives_reranking(index):
    """A known-item lookup must not be reordered by a citation-blind model."""
    conn, emb = index
    query = f"what does {DISCLOSURE} require"
    hits = _hits(conn, emb, query)
    assert hits[0].pinned and hits[0].section_citation == DISCLOSURE

    # A reranker that loves every *other* section still cannot displace it.
    hostile = FakeReranker(boost={"travel": 100.0, "the": 50.0})
    out = rr.rerank(query, hits, k=3, reranker=hostile)

    assert out[0].section_citation == DISCLOSURE
    # ...and it was never handed to the cross-encoder at all.
    assert all(DISCLOSURE not in p for p in hostile.calls[0][1])


def test_unpinned_results_are_not_flagged(index):
    conn, emb = index
    hits = _hits(conn, emb, "travel reimbursement expense")
    assert not any(h.pinned for h in hits)


def test_rerank_with_only_pinned_results_skips_the_model(index):
    conn, emb = index
    hits = [h for h in _hits(conn, emb, f"what does {DISCLOSURE} require") if h.pinned]
    reranker = FakeReranker()
    out = rr.rerank("q", hits, k=5, reranker=reranker)
    assert [r.section_citation for r in out] == [h.section_citation for h in hits]
    assert reranker.calls == []


# --- rerank_search end-to-end ----------------------------------------------


def test_rerank_search_returns_k_distinct_sections(index):
    conn, emb = index
    out = rr.rerank_search(
        "financial disclosure gao employee",
        2,
        conn=conn,
        embedder=emb,
        reranker=FakeReranker(),
        candidates=10,
    )
    assert len(out) == 2
    assert len({r.section_citation for r in out}) == 2


def test_rerank_search_candidates_floor_is_k(index):
    """Asking for more results than `candidates` still returns k, not `candidates`."""
    conn, emb = index
    reranker = FakeReranker()
    out = rr.rerank_search(
        "financial disclosure",
        3,
        conn=conn,
        embedder=emb,
        reranker=reranker,
        candidates=1,
    )
    assert len(out) == 3
    assert len(reranker.calls[0][1]) >= 3
