"""Tests for the gold-set gate and draft scaffolding (eval.goldset)."""

from __future__ import annotations

import pytest

from eval import goldset
from index import store
from pipeline.manifest import TitleSpec
from tests.test_index import FIXTURE_CHUNKS, FakeEmbedder

SPEC4 = TitleSpec(4, "2024-07-18", "Accounts")
SPEC31 = TitleSpec(31, "2024-07-18")
SECTIONS = {"4 CFR § 7.3", "4 CFR § 8.1"}


def _checks(problems):
    return {p.check for p in problems}


def _entry(**kw):
    base = {
        "question": "a question",
        "relevant_citations": ["4 CFR § 7.3"],
        "kind": "prose",
        "status": "approved",
    }
    base.update(kw)
    return base


# --- validation -------------------------------------------------------------


def test_clean_entry_validates():
    assert validate_only([_entry()]) == set()


def validate_only(entries, **kw):
    return _checks(goldset.validate(entries, sections=SECTIONS, **kw))


def test_empty_question_is_an_error():
    """An unfilled draft stub must never be mistaken for a finished entry."""
    assert "gold:empty-question" in validate_only([_entry(question="  ")])


def test_duplicate_question_is_an_error():
    """A duplicate silently double-weights one question in every metric."""
    assert "gold:duplicate" in validate_only([_entry(), _entry()])


def test_unknown_citation_is_an_error():
    assert "gold:unknown-citation" in validate_only([_entry(relevant_citations=["4 CFR § 9.9"])])


def test_bad_kind_and_status_are_errors():
    problems = validate_only([_entry(kind="lookup", status="maybe")])
    assert {"gold:kind", "gold:status"} <= problems


def test_citations_must_be_a_list():
    assert "gold:citations" in validate_only([_entry(relevant_citations=None)])


def test_negative_without_a_corpus_record_warns():
    problems = goldset.validate(
        [_entry(relevant_citations=[])], sections=SECTIONS, specs=[SPEC4]
    )
    assert "gold:negative-unverified" in _checks(problems)


def test_negative_verified_against_a_smaller_corpus_is_stale():
    """A negative is only negative with respect to a corpus; growing it
    silently turns some negatives into answerable questions."""
    entry = _entry(relevant_citations=[], negative_corpus=["4@2024-07-18"])
    problems = goldset.validate([entry], sections=SECTIONS, specs=[SPEC4, SPEC31])
    stale = [p for p in problems if p.check == "gold:negative-stale"]
    assert stale and "31@2024-07-18" in stale[0].message


def test_negative_verified_against_the_current_corpus_is_clean():
    entry = _entry(relevant_citations=[], negative_corpus=["4@2024-07-18"])
    problems = goldset.validate([entry], sections=SECTIONS, specs=[SPEC4])
    assert "gold:negative-stale" not in _checks(problems)


def test_negative_citing_an_indexed_section_is_an_error():
    """Not merely stale — simply wrong: the section it names is in the index."""
    entry = _entry(
        question="What are the recordkeeping requirements in 4 CFR 7.3?",
        relevant_citations=[],
        kind="citation",
        negative_corpus=["4@2024-07-18"],
    )
    problems = goldset.validate([entry], sections=SECTIONS, specs=[SPEC4])
    assert "gold:negative-answerable" in _checks(problems)


def test_negative_citing_an_absent_section_is_fine():
    entry = _entry(
        question="What are the recordkeeping requirements in 4 CFR 4.999?",
        relevant_citations=[],
        kind="citation",
        negative_corpus=["4@2024-07-18"],
    )
    problems = goldset.validate([entry], sections=SECTIONS, specs=[SPEC4])
    assert "gold:negative-answerable" not in _checks(problems)


# --- coverage ---------------------------------------------------------------


def test_coverage_counts_by_title_and_kind():
    entries = [
        _entry(question="p1"),
        _entry(question="p2"),
        _entry(question="c1", kind="citation"),
        _entry(question="n1", relevant_citations=[]),
        _entry(question="draft", status="draft"),
    ]
    cov = goldset.coverage(entries, [SPEC4])
    assert cov.per_title[4] == {"prose": 2, "citation": 1}
    assert cov.negatives == 1, "drafts must not count toward coverage"


def test_coverage_gate_flags_an_unmeasured_title():
    cov = goldset.coverage([], [SPEC4])
    problems = check_cov(cov, [SPEC4])
    assert {"gold:coverage-prose", "gold:coverage-citation"} <= problems


def check_cov(cov, specs):
    return _checks(goldset.check_coverage(cov, specs))


def test_coverage_gate_passes_when_the_floor_is_met():
    entries = (
        [_entry(question=f"p{i}") for i in range(goldset.MIN_PROSE_PER_TITLE)]
        + [_entry(question=f"c{i}", kind="citation") for i in range(goldset.MIN_CITATION_PER_TITLE)]
        + [
            _entry(question=f"n{i}", relevant_citations=[])
            for i in range(goldset.MIN_NEGATIVES_PER_TITLE)
        ]
    )
    cov = goldset.coverage(entries, [SPEC4])
    assert goldset.check_coverage(cov, [SPEC4]) == []


def test_negatives_requirement_scales_with_the_corpus():
    entries = [
        _entry(question=f"n{i}", relevant_citations=[])
        for i in range(goldset.MIN_NEGATIVES_PER_TITLE)
    ]
    cov = goldset.coverage(entries, [SPEC4, SPEC31])
    assert "gold:coverage-negatives" in check_cov(cov, [SPEC4, SPEC31])


# --- draft scaffolding ------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "index" / "regsearch.db"
    store.build_index(FIXTURE_CHUNKS, db_path=db_path, embedder=FakeEmbedder(), show_progress=False)
    return db_path


def test_draft_generates_answerable_citation_lookups(db):
    entries = goldset.draft_for_title(SPEC4, db_path=db, n_prose=0, n_citation=2, n_negatives=0)
    assert len(entries) == 2
    for e in entries:
        assert e["kind"] == "citation"
        assert e["relevant_citations"], "a generated lookup must name its answer"
        assert e["relevant_citations"][0] in e["question"]
        assert e["status"] == "draft"


def test_draft_generates_negatives_naming_an_absent_section(db):
    entries = goldset.draft_for_title(SPEC4, db_path=db, n_prose=0, n_citation=0, n_negatives=1)
    present = {c["section_citation"] for c in FIXTURE_CHUNKS}
    for e in entries:
        assert e["relevant_citations"] == []
        assert e["negative_corpus"]
        assert not any(sec.split(" § ")[-1] in e["question"] for sec in present)


def test_draft_prose_stubs_have_no_question(db):
    """Prose questions cannot be generated honestly, so they are left blank
    and the validator rejects them until a human writes one."""
    entries = goldset.draft_for_title(SPEC4, db_path=db, n_prose=2, n_citation=0, n_negatives=0)
    assert entries and all(e["question"] == "" for e in entries)
    assert all("TODO" in e["notes"] for e in entries)
    assert all(e["relevant_citations"] for e in entries)
    assert "gold:empty-question" in validate_only(entries)


def test_draft_is_deterministic(db):
    a = goldset.draft_for_title(SPEC4, db_path=db, n_prose=2, n_citation=2, n_negatives=1)
    b = goldset.draft_for_title(SPEC4, db_path=db, n_prose=2, n_citation=2, n_negatives=1)
    assert a == b


def test_draft_counts_are_independent(db):
    """Changing one category must not reshuffle an already-reviewed other."""
    a = goldset.draft_for_title(SPEC4, db_path=db, n_prose=1, n_citation=2, n_negatives=0)
    b = goldset.draft_for_title(SPEC4, db_path=db, n_prose=2, n_citation=2, n_negatives=0)
    cites_a = [e for e in a if e["kind"] == "citation"]
    cites_b = [e for e in b if e["kind"] == "citation"]
    assert cites_a == cites_b


def test_draft_for_an_unindexed_title_fails_clearly(db):
    with pytest.raises(ValueError, match="index it first"):
        goldset.draft_for_title(SPEC31, db_path=db)


# --- round trip -------------------------------------------------------------


def test_write_and_load_round_trip(tmp_path):
    entries = [_entry(question="q1"), _entry(question="q2")]
    path = goldset.write(entries, tmp_path / "g.jsonl")
    assert goldset.load(path) == entries


def test_load_reports_the_offending_line(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"question": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2"):
        goldset.load(p)
