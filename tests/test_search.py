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


class TestBm25Confidence:
    """Contrast-based gating of a BM25 ranking (see search.bm25_confidence)."""

    def test_flat_ranking_is_noise(self):
        # Nothing stands out: every chunk scores about the same.
        assert search.bm25_confidence([10.0] * 20) == 0.0

    def test_peaked_ranking_is_trusted(self):
        # The top hit towers over rank 10 — a real lexical match.
        assert search.bm25_confidence([50.0] + [10.0] * 19) == 1.0

    def test_ramps_between_thresholds(self):
        lo, hi = search.BM25_CONTRAST_LO, search.BM25_CONTRAST_HI
        midpoint = (lo + hi) / 2
        scores = [midpoint] + [1.0] * 19
        assert search.bm25_confidence(scores) == pytest.approx(0.5)

    def test_contrast_is_scale_free(self):
        """Doubling every score (e.g. a longer query) must not change trust."""
        scores = [3.0] + [1.0] * 19
        assert search.bm25_confidence([2 * s for s in scores]) == search.bm25_confidence(scores)

    def test_few_matches_is_discriminative_not_noise(self):
        # Only three chunks share any term with the query at all.
        assert search.bm25_confidence([9.0, 8.0, 7.0]) == 1.0

    def test_no_matches(self):
        assert search.bm25_confidence([]) == 0.0
        assert search.bm25_confidence([0.0] * 20) == 0.0


def test_noisy_bm25_does_not_displace_the_top_vector_hit(index):
    """The regression that made plain RRF score below vector-only search.

    With a zero-confidence BM25 ranking, hybrid must fall back to semantic
    order rather than letting arbitrary keyword hits crowd the top.
    """
    conn, emb = index
    query = "financial disclosure gao employee file"
    vector_only = search.vector_search(query, 4, conn=conn, embedder=emb)
    hybrid = search.hybrid_search(query, 4, conn=conn, embedder=emb, bm25_weight=0.0)
    assert [r.section_citation for r in hybrid] == [r.section_citation for r in vector_only]
    assert hybrid[0].citation == DISCLOSURE


class TestExactCitationLookup:
    """Naming a section is a known-item lookup, not a similarity search."""

    def test_citation_query_returns_that_section_first(self, index):
        conn, emb = index
        for query in ("§ 8.1", "4 CFR 8.1", "section 8.1"):
            top = search.hybrid_search(query, 4, conn=conn, embedder=emb)[0]
            assert top.section_citation == TRAVEL, query

    def test_prose_queries_are_unaffected(self, index):
        """No citation in the query means no exact-match promotion at all."""
        conn, emb = index
        results = search.hybrid_search(
            "financial disclosure gao employee file", 4, conn=conn, embedder=emb
        )
        assert results[0].citation == DISCLOSURE

    def test_citation_in_query_overrides_the_confidence_gate(self, index, monkeypatch):
        """BM25 must not be gated off when the query names a section.

        The contrast heuristic guesses whether BM25 found a real lexical
        anchor; a citation in the query answers that directly. Pinned because
        a flat-looking score curve silently discarded the one ranking that
        knew where a cited section was referenced.
        """
        conn, emb = index
        monkeypatch.setattr(search, "bm25_confidence", lambda *a, **kw: 0.0)
        results = search.hybrid_search("§ 8.1", 4, conn=conn, embedder=emb)
        assert any("bm25" in r.retrievers for r in results)

    def test_a_number_in_prose_is_not_a_citation_lookup(self, index):
        """"8.1" alone reads as a section, but "8 hours" must not."""
        assert search.citation_tokens("how many hours of annual leave") == set()
        assert "§8.1" in search.citation_tokens("what does § 8.1 say")


def test_hybrid_returns_one_result_per_section(index):
    conn, emb = index
    results = search.hybrid_search("employee gao report", 4, conn=conn, embedder=emb)
    sections = [r.section_citation for r in results]
    assert len(sections) == len(set(sections))


def test_results_carry_pure_text_and_citation(index):
    conn, emb = index
    results = search.vector_search("ethics conflict", 2, conn=conn, embedder=emb)
    top = results[0]
    assert top.citation.startswith("4 CFR")
    assert "Title 4" not in top.text  # pure body only
    assert top.heading_path[0] == "Title 4—Accounts"
