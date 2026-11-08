"""Tests for api.runs: the read-only view over eval/results.

Two things are worth guarding here. First, the dashboard must agree with the
gate — if ``eval.compare`` says the gold set changed and the deltas are not
evidence, the dashboard must not display a regression. Second, the
per-question matrix must not read a negative question's absent rank as a miss:
a negative has no correct section by construction, so "not found" is the
definition of the question, not a result about the retriever.
"""

from __future__ import annotations

import json

import pytest

from api import runs as runs_mod
from api.runs import (
    MISS,
    NEGATIVE,
    TOP5,
    TOP10,
    compare_runs,
    list_runs,
    load_run,
    question_deltas,
    status_matrix,
)

MODES = ("vector", "bm25", "hybrid")


def make_summary(recall10=0.9, n=54):
    return {
        m: {"recall@5": 0.87, "recall@10": recall10, "mrr@10": 0.71, "n": n}
        for m in MODES
    }


def make_detail(question, citations, ranks, kind="prose"):
    return {
        "question": question,
        "relevant_citations": list(citations),
        "kind": kind,
        "modes": {
            m: {"first_hit_rank": ranks.get(m), "top_sections": [f"{m} CFR § 1.{i}" for i in
                                                                 range(3)]}
            for m in MODES
        },
    }


def make_run(stamp, *, recall10=0.9, details=None, gold_sha="aaaa", gold_n=54, commit="abc1234"):
    return {
        "timestamp": stamp,
        "git_commit": commit,
        "k": 10,
        "modes": list(MODES),
        "n_in_corpus": gold_n,
        "n_negatives": 6,
        "rerank_model": None,
        "params": {"rrf_k": 10},
        "gold_fingerprint": {"n": gold_n, "sha256": gold_sha},
        "summary": make_summary(recall10, gold_n),
        "summary_by_kind": {
            "prose": {m: {**v, "n": 42} for m, v in make_summary(recall10, gold_n).items()},
            "citation": {m: {**v, "n": 12} for m, v in make_summary(recall10, gold_n).items()},
        },
        "details": details if details is not None else [],
        "_file": f"{stamp}.json",
    }


@pytest.fixture
def results_dir(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    for stamp, recall in (("20260101-000000", 0.90), ("20260102-000000", 0.85)):
        run = make_run(stamp, recall10=recall)
        run.pop("_file")
        (out / f"{stamp}.json").write_text(json.dumps(run), encoding="utf-8")
    return out


# --- loading ----------------------------------------------------------------


class TestLoading:
    def test_lists_newest_first(self, results_dir):
        metas = list_runs(results_dir)
        assert [m.timestamp for m in metas] == ["20260102-000000", "20260101-000000"]
        assert metas[0].gold_n == 54 and metas[0].git_commit == "abc1234"

    def test_loads_one_run_by_name(self, results_dir):
        assert load_run("20260101-000000.json", results_dir)["timestamp"] == "20260101-000000"

    @pytest.mark.parametrize(
        "name", ["../../etc/passwd", "/etc/passwd", "..%2Fx.json", "no-such.json", "x.txt"]
    )
    def test_refuses_anything_that_is_not_a_run_in_the_directory(self, name, results_dir):
        """``name`` comes from a URL path segment."""
        with pytest.raises((ValueError, FileNotFoundError)):
            load_run(name, results_dir)

    def test_symlink_out_of_the_directory_is_refused(self, results_dir, tmp_path):
        secret = tmp_path / "secret.json"
        secret.write_text('{"summary": {}}', encoding="utf-8")
        try:
            (results_dir / "link.json").symlink_to(secret)
        except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
            pytest.skip("symlinks unavailable")
        with pytest.raises(FileNotFoundError):
            load_run("link.json", results_dir)

    def test_empty_directory_has_no_latest_comparison(self, tmp_path):
        assert runs_mod.latest_comparison(tmp_path) is None


# --- comparison -------------------------------------------------------------


class TestComparison:
    def test_deltas_match_eval_compare(self, results_dir):
        comparison = runs_mod.latest_comparison(results_dir)
        delta = next(
            d for d in comparison.deltas
            if (d.group, d.mode, d.metric) == (runs_mod.ALL, "hybrid", "recall@10")
        )
        assert delta.change == pytest.approx(-0.05)
        assert delta in comparison.regressions  # 0.05 > 1.5/54

    def test_a_changed_gold_set_suppresses_the_gate_but_keeps_the_numbers(self):
        """Means over different question sets are not a regression, they are a different
        measurement wearing the same name."""
        current = make_run("b", recall10=0.60, gold_sha="bbbb", gold_n=60)
        reference = make_run("a", recall10=0.90, gold_sha="aaaa", gold_n=54)
        comparison = compare_runs(current, reference)
        assert not comparison.comparable
        assert comparison.regressions == []
        assert comparison.deltas, "the deltas are still shown"
        assert "question set changed" in comparison.note

    def test_no_reference_is_not_an_error(self):
        comparison = compare_runs(make_run("a"), None)
        assert comparison.deltas == [] and comparison.regressions == []

    def test_groups_put_all_first(self, results_dir):
        comparison = runs_mod.latest_comparison(results_dir)
        assert comparison.groups[0] == runs_mod.ALL


# --- the per-question matrix ------------------------------------------------


class TestStatusMatrix:
    def rows(self, **kw):
        details = [
            make_detail("found first", ["4 CFR § 2.3"], {"vector": 1, "bm25": 2, "hybrid": 1}),
            make_detail("deep", ["4 CFR § 2.6"], {"vector": 8, "bm25": None, "hybrid": 7}),
            make_detail("missed", ["4 CFR § 3.1"], {}),
            make_detail("no answer exists", [], {}, kind="prose"),
            make_detail("cite", ["31 CFR § 1.1"], {"vector": 1, "bm25": 1, "hybrid": 1},
                        kind="citation"),
        ]
        return status_matrix(make_run("a", details=details), modes=MODES, **kw)

    def test_classifies_ranks_into_bands(self):
        by_q = {r.question: r for r in self.rows()}
        assert by_q["found first"].states["hybrid"] == TOP5
        assert by_q["deep"].states["hybrid"] == TOP10
        assert by_q["deep"].states["bm25"] == MISS
        assert by_q["missed"].states["vector"] == MISS

    def test_a_negative_is_not_a_miss(self):
        """It has no relevant citation, so no mode can have a rank. Marking that a
        miss would read a guarantee as a result."""
        row = next(r for r in self.rows() if r.question == "no answer exists")
        assert row.negative
        assert set(row.states.values()) == {NEGATIVE}

    def test_worst_questions_come_first_and_negatives_sink(self):
        order = [r.question for r in self.rows()]
        assert order[0] == "missed"
        assert order[-1] == "no answer exists"

    def test_filters_by_kind(self):
        assert [r.question for r in self.rows(kind="citation")] == ["cite"]

    def test_only_problems_hides_the_clean_rows(self):
        shown = {r.question for r in self.rows(only_problems=True)}
        assert shown == {"deep", "missed"}
        assert "found first" not in shown and "no answer exists" not in shown

    def test_keeps_the_sections_each_mode_returned(self):
        row = next(r for r in self.rows() if r.question == "found first")
        assert row.top["hybrid"][0].startswith("hybrid CFR")


# --- per-question movement --------------------------------------------------


class TestQuestionDeltas:
    def pair(self):
        reference = make_run("a", details=[
            make_detail("stayed", ["x"], {"hybrid": 1}),
            make_detail("improved", ["x"], {"hybrid": 9}),
            make_detail("worsened", ["x"], {"hybrid": 2}),
            make_detail("broke", ["x"], {"hybrid": 3}),
            make_detail("fixed", ["x"], {"hybrid": None}),
            make_detail("negative", [], {}),
            make_detail("retired", ["x"], {"hybrid": 1}),
        ])
        current = make_run("b", details=[
            make_detail("stayed", ["x"], {"hybrid": 1}),
            make_detail("improved", ["x"], {"hybrid": 2}),
            make_detail("worsened", ["x"], {"hybrid": 6}),
            make_detail("broke", ["x"], {"hybrid": None}),
            make_detail("fixed", ["x"], {"hybrid": 4}),
            make_detail("negative", [], {}),
            make_detail("brand new", ["x"], {"hybrid": 1}),
        ])
        return current, reference

    def test_classifies_every_movement(self):
        current, reference = self.pair()
        moved = {d.question: d.movement for d in question_deltas(current, reference,
                                                                 mode="hybrid")}
        assert moved == {
            "broke": "broken", "worsened": "worse", "fixed": "fixed", "improved": "better"
        }

    def test_bad_news_first(self):
        current, reference = self.pair()
        assert [d.movement for d in question_deltas(current, reference, mode="hybrid")][:2] == [
            "broken", "worse"
        ]

    def test_skips_questions_not_in_both_runs(self):
        current, reference = self.pair()
        questions = {d.question for d in question_deltas(current, reference, mode="hybrid")}
        assert "brand new" not in questions and "retired" not in questions

    def test_negatives_have_no_rank_to_move(self):
        current, reference = self.pair()
        moved = question_deltas(current, reference, mode="hybrid", include_same=True)
        assert "negative" not in {d.question for d in moved}

    def test_unchanged_questions_are_opt_in(self):
        current, reference = self.pair()
        assert "stayed" not in {d.question for d in question_deltas(current, reference,
                                                                    mode="hybrid")}
        assert "stayed" in {
            d.question
            for d in question_deltas(current, reference, mode="hybrid", include_same=True)
        }
