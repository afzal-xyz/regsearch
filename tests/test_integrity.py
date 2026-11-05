"""Tests for the integrity gate (pipeline.integrity).

Each test breaks the corpus in exactly one way and asserts the gate names it.
A gate nobody has seen fail is a gate nobody knows works.
"""

from __future__ import annotations

import json

import pytest

from index import store
from pipeline import integrity
from pipeline.manifest import TitleSpec
from tests.test_index import FIXTURE_CHUNKS, FakeEmbedder

SPEC = TitleSpec(4, "2024-07-18", "Accounts")


def _checks(problems) -> set[str]:
    return {p.check for p in problems}


def _errors(problems):
    return [p for p in problems if p.severity == integrity.ERROR]


# --- chunk-level ------------------------------------------------------------


def test_clean_chunks_pass():
    assert integrity.check_chunk_file(SPEC, FIXTURE_CHUNKS) == []


def test_empty_chunk_list_fails():
    problems = integrity.check_chunk_file(SPEC, [])
    assert _checks(problems) == {"chunk:empty"}


def test_empty_text_is_caught():
    chunks = [dict(c) for c in FIXTURE_CHUNKS]
    chunks[0]["text"] = "   "
    assert "chunk:empty-text" in _checks(integrity.check_chunk_file(SPEC, chunks))


def test_missing_required_field_is_caught():
    chunks = [dict(c) for c in FIXTURE_CHUNKS]
    del chunks[0]["section_citation"]
    problems = integrity.check_chunk_file(SPEC, chunks)
    assert "chunk:missing-section_citation" in _checks(problems)


def test_malformed_citation_is_caught():
    chunks = [dict(c) for c in FIXTURE_CHUNKS]
    chunks[0]["citation"] = "section 7.3"
    assert "chunk:citation-format" in _checks(integrity.check_chunk_file(SPEC, chunks))


@pytest.mark.parametrize(
    "citation",
    [
        "4 CFR § 7.3",
        "4 CFR § 1010.311(a)(2)",
        "4 CFR Appendix I to Part 83",
        "4 CFR Appendix A to Subpart C of Part 1010",
        "4 CFR Appendix to Part 50",
        "4 CFR Appendix A to Chapter V",
    ],
)
def test_real_citation_shapes_are_accepted(citation):
    """Appendices are real regulation text, not malformed sections.

    Two earlier drafts of this check rejected Title 4's and Title 31's
    appendices in turn; these are the shapes the corpus actually contains.
    """
    chunks = [dict(FIXTURE_CHUNKS[0])]
    chunks[0]["citation"] = citation
    chunks[0]["section_citation"] = citation
    assert "chunk:citation-format" not in _checks(integrity.check_chunk_file(SPEC, chunks))


def test_citation_from_another_title_is_caught():
    chunks = [dict(c) for c in FIXTURE_CHUNKS]
    chunks[0]["citation"] = "31 CFR § 7.3"
    chunks[0]["section_citation"] = "31 CFR § 7.3"
    assert "chunk:title-mismatch" in _checks(integrity.check_chunk_file(SPEC, chunks))


def test_wrong_date_is_caught():
    chunks = [dict(c) for c in FIXTURE_CHUNKS]
    chunks[0]["date"] = "2025-01-01"
    assert "chunk:date-mismatch" in _checks(integrity.check_chunk_file(SPEC, chunks))


def test_duplicate_chunk_id_is_caught():
    """Duplicate ids make indexing silently drop chunks (chunk_id is UNIQUE)."""
    chunks = [dict(c) for c in FIXTURE_CHUNKS]
    chunks[1]["chunk_id"] = chunks[0]["chunk_id"]
    assert "chunk:duplicate-id" in _checks(integrity.check_chunk_file(SPEC, chunks))


# --- index-level ------------------------------------------------------------


@pytest.fixture
def index(tmp_path):
    db_path = tmp_path / "index" / "regsearch.db"
    store.build_index(FIXTURE_CHUNKS, db_path=db_path, embedder=FakeEmbedder(), show_progress=False)
    conn = store.connect(db_path)
    yield conn
    conn.close()


def test_clean_index_passes(index):
    counts = {4: len(FIXTURE_CHUNKS)}
    assert integrity.check_index(index, [SPEC], counts) == []


def test_declared_title_absent_from_index_is_caught(index):
    other = TitleSpec(31, "2024-07-18")
    problems = integrity.check_index(index, [SPEC, other], {4: len(FIXTURE_CHUNKS)})
    assert "index:missing-title" in _checks(problems)


def test_undeclared_title_in_index_is_caught(index):
    """A title nobody declared is searched but never evaluated."""
    index.execute("update chunks set title = 99 where id = (select min(id) from chunks)")
    problems = integrity.check_index(index, [SPEC], {4: len(FIXTURE_CHUNKS)})
    assert "index:undeclared-title" in _checks(problems)


def test_date_mismatch_against_manifest_is_caught(index):
    problems = integrity.check_index(index, [TitleSpec(4, "2025-01-01")], {})
    assert "index:date-mismatch" in _checks(problems)


def test_stale_index_count_is_caught(index):
    """More chunks on disk than in the index means the index stage didn't rerun."""
    problems = integrity.check_index(index, [SPEC], {4: len(FIXTURE_CHUNKS) + 5})
    assert "index:count-mismatch" in _checks(problems)


def test_missing_vectors_are_caught(index):
    index.execute("delete from vec_chunks where rowid = (select min(rowid) from vec_chunks)")
    problems = integrity.check_index(index, [SPEC], {4: len(FIXTURE_CHUNKS)})
    assert "index:vector-count" in _checks(problems)


def test_stale_analyzer_version_warns(index, monkeypatch):
    """A stale BM25 index self-heals at query time, but the confidence
    thresholds were calibrated against the old analyzer — worth saying so."""
    import pickle

    row = index.execute("select data from bm25 where id = 1").fetchone()
    payload = pickle.loads(row["data"])
    payload["analyzer_version"] = integrity.ANALYZER_VERSION - 1
    index.execute("update bm25 set data = ? where id = 1", (pickle.dumps(payload),))

    assert integrity.stored_analyzer_version(index) == integrity.ANALYZER_VERSION - 1
    problems = integrity.check_index(index, [SPEC], {4: len(FIXTURE_CHUNKS)})
    analyzer = [p for p in problems if p.check == "index:analyzer"]
    assert analyzer and analyzer[0].severity == integrity.WARNING


# --- retrieval probe --------------------------------------------------------


def test_retrieval_probe_passes_on_a_working_index(index):
    assert integrity.check_retrieval(index, [SPEC], n_probe=5) == []


def test_retrieval_probe_catches_a_dead_bm25_index(index, monkeypatch):
    """Every structural check can pass on an index that finds nothing."""
    monkeypatch.setattr("retrieval.search.bm25_search", lambda *a, **k: [])
    problems = integrity.check_retrieval(index, [SPEC], n_probe=5)
    assert "retrieval:self-lookup" in _checks(problems)
    assert _errors(problems)


def test_retrieval_probe_is_deterministic(index):
    a = integrity.check_retrieval(index, [SPEC], n_probe=3)
    b = integrity.check_retrieval(index, [SPEC], n_probe=3)
    assert [str(p) for p in a] == [str(p) for p in b]


# --- gate -------------------------------------------------------------------


def test_run_checks_reports_a_missing_chunk_file(tmp_path):
    problems = integrity.run_checks(
        [SPEC], db_path=tmp_path / "none.db", chunks_dir=tmp_path, probe=False
    )
    assert "chunk:missing-file" in _checks(problems)
    assert "index:missing" in _checks(problems)


def test_run_checks_passes_on_a_consistent_corpus(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    path = SPEC.chunks_path(chunks_dir)
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in FIXTURE_CHUNKS), encoding="utf-8"
    )
    db_path = tmp_path / "index" / "regsearch.db"
    store.build_index(FIXTURE_CHUNKS, db_path=db_path, embedder=FakeEmbedder(), show_progress=False)

    problems = integrity.run_checks([SPEC], db_path=db_path, chunks_dir=chunks_dir, params=False)
    assert problems == [], [str(p) for p in problems]


def test_summarize_strict_fails_on_warnings():
    warn = [integrity.Problem("x", integrity.WARNING, "m")]
    assert integrity.summarize(warn, strict=False)[0] is True
    assert integrity.summarize(warn, strict=True)[0] is False


def test_summarize_always_fails_on_errors():
    err = [integrity.Problem("x", integrity.ERROR, "m")]
    assert integrity.summarize(err, strict=False)[0] is False
