"""Tests for the eval harness: pure metric math + an integration run on the
FakeEmbedder fixture index (no model download)."""

from __future__ import annotations

import pytest

from eval import run
from index import store
from tests.test_index import FIXTURE_CHUNKS, FakeEmbedder

# --- pure metric math (hand-computed) --------------------------------------

RANKED = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


def test_first_hit_rank():
    assert run.first_hit_rank(RANKED, {"a"}) == 1
    assert run.first_hit_rank(RANKED, {"c"}) == 3
    assert run.first_hit_rank(RANKED, {"z"}) is None
    assert run.first_hit_rank(RANKED, {"h", "c"}) == 3  # earliest of the two


def test_recall_at_k_boundary():
    # relevant sits at rank 7 -> in top-10 but not top-5
    assert run.recall_at_k(RANKED, {"g"}, 5) == 0.0
    assert run.recall_at_k(RANKED, {"g"}, 10) == 1.0
    # rank 1 -> in both
    assert run.recall_at_k(RANKED, {"a"}, 5) == 1.0
    # absent -> 0
    assert run.recall_at_k(RANKED, {"z"}, 10) == 0.0


def test_reciprocal_rank():
    assert run.reciprocal_rank_at_k(RANKED, {"a"}, 10) == 1.0
    assert run.reciprocal_rank_at_k(RANKED, {"b"}, 10) == 0.5
    assert run.reciprocal_rank_at_k(RANKED, {"c"}, 10) == pytest.approx(1 / 3)
    # hit at rank 7 is outside a top-5 window -> 0
    assert run.reciprocal_rank_at_k(RANKED, {"g"}, 5) == 0.0
    assert run.reciprocal_rank_at_k(RANKED, {"g"}, 10) == pytest.approx(1 / 7)
    # miss -> 0
    assert run.reciprocal_rank_at_k(RANKED, {"z"}, 10) == 0.0


def test_negatives_are_undefined_not_zero():
    # empty relevant set: metrics are undefined (None), not 0.0
    assert run.recall_at_k(RANKED, set(), 10) is None
    assert run.reciprocal_rank_at_k(RANKED, set(), 10) is None
    assert run.first_hit_rank(RANKED, set()) is None


# --- integration on the fixture index --------------------------------------

GOLD = [
    {
        "question": "financial disclosure employee file report",
        "relevant_citations": ["4 CFR § 7.3"],
        "status": "approved",
    },
    {
        "question": "travel reimbursement expense",
        "relevant_citations": ["4 CFR § 8.1"],
        "status": "approved",
    },
    {
        "question": "something totally unrelated to this corpus",
        "relevant_citations": [],
        "status": "approved",
    },
]


@pytest.fixture
def index(tmp_path):
    db_path = tmp_path / "index" / "regsearch.db"
    emb = FakeEmbedder()
    store.build_index(FIXTURE_CHUNKS, db_path=db_path, embedder=emb, show_progress=False)
    conn = store.connect(db_path)
    yield conn, emb
    conn.close()


def test_evaluate_end_to_end(index):
    conn, emb = index
    result = run.evaluate(GOLD, ["vector", "bm25", "hybrid"], conn=conn, embedder=emb, k=10)

    assert result["n_in_corpus"] == 2
    assert result["n_negatives"] == 1

    # Hybrid should find both in-corpus answers within the top results.
    hybrid = result["summary"]["hybrid"]
    assert hybrid["recall@10"] == 1.0
    assert hybrid["mrr@10"] > 0.0
    assert hybrid["n"] == 2


def test_details_capture_per_mode_first_hit(index):
    conn, emb = index
    result = run.evaluate(GOLD, ["vector", "bm25", "hybrid"], conn=conn, embedder=emb, k=10)
    by_q = {d["question"]: d for d in result["details"]}

    # the disclosure question is hit by hybrid at some concrete rank
    disc = by_q["financial disclosure employee file report"]
    assert disc["modes"]["hybrid"]["first_hit_rank"] is not None

    # the negative has null first-hit rank in every mode (nothing is "relevant")
    neg = by_q["something totally unrelated to this corpus"]
    assert all(neg["modes"][m]["first_hit_rank"] is None for m in ("vector", "bm25", "hybrid"))


def test_retrieve_returns_distinct_sections(index):
    conn, emb = index
    sections = run.retrieve("bm25", "employee", 10, conn=conn, embedder=emb)
    assert len(sections) == len(set(sections))
