"""Tests for retrieval.search against a tiny fixture index (fake embedder)."""

from __future__ import annotations

import pytest

from index import store
from index.store import ModelMismatchError
from retrieval import search
from tests.test_index import FIXTURE_CHUNKS, FakeEmbedder

DISCLOSURE = "4 CFR § 7.3"
TRAVEL = "4 CFR § 8.1"


@pytest.fixture
def index(tmp_path):
    db_path = tmp_path / "index" / "regsearch.db"
    emb = FakeEmbedder()
    store.build_index(FIXTURE_CHUNKS, db_path=db_path, embedder=emb, show_progress=False)
    conn = store.connect(db_path)
    yield conn, emb
    conn.close()


def test_vector_search_finds_disclosure(index):
    conn, emb = index
    results = search.vector_search(
        "financial disclosure gao employee file", 3, conn=conn, embedder=emb
    )
    assert results
    assert results[0].citation == DISCLOSURE
    assert results[0].retrievers == ["vector"]


def test_bm25_search_finds_keyword_match(index):
    conn, _ = index
    results = search.bm25_search("travel reimbursement expense", 3, conn=conn)
    assert results
    assert results[0].citation == TRAVEL
    assert results[0].retrievers == ["bm25"]


def test_hybrid_merges_and_reports_retrievers(index):
    conn, emb = index
    results = search.hybrid_search(
        "when must a GAO employee file a financial disclosure", 3, conn=conn, embedder=emb
    )
    assert 0 < len(results) <= 3
    assert results[0].citation == DISCLOSURE
    # scores are descending (RRF)
    assert all(a.score >= b.score for a, b in zip(results, results[1:], strict=False))
    # both retrievers are represented across the merged results
    seen = {r for res in results for r in res.retrievers}
    assert "vector" in seen and "bm25" in seen
    # the top hit was surfaced by both retrievers
    assert set(results[0].retrievers) == {"bm25", "vector"}


def test_hybrid_respects_k(index):
    conn, emb = index
    assert len(search.hybrid_search("employee", 2, conn=conn, embedder=emb)) <= 2


def test_query_model_mismatch_fails_loudly(index):
    conn, _ = index
    wrong = FakeEmbedder("a-different-model")
    with pytest.raises(ModelMismatchError):
        search.vector_search("financial disclosure", 3, conn=conn, embedder=wrong)
    with pytest.raises(ModelMismatchError):
        search.hybrid_search("financial disclosure", 3, conn=conn, embedder=wrong)


def test_results_carry_pure_text_and_citation(index):
    conn, emb = index
    results = search.vector_search("ethics conflict", 2, conn=conn, embedder=emb)
    top = results[0]
    assert top.citation.startswith("4 CFR")
    assert "Title 4" not in top.text  # pure body only
    assert top.heading_path[0] == "Title 4—Accounts"
