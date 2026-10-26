"""Tests for eval.compare: run diffing, the regression threshold, and the report."""

from __future__ import annotations

import json

import pytest

from eval import compare
from eval.compare import ALL, Delta, diff_runs, load_runs, pick_reference


def make_run(name, hybrid_recall10=0.9, *, prose_n=42, citation=True, commit="abc1234"):
    """A minimal results payload in the shape eval.run writes."""
    modes = {
        "vector": {"recall@5": 0.78, "recall@10": 0.87, "mrr@10": 0.63, "n": prose_n + 12},
        "bm25": {"recall@5": 0.70, "recall@10": 0.76, "mrr@10": 0.55, "n": prose_n + 12},
        "hybrid": {"recall@5": 0.87, "recall@10": hybrid_recall10, "mrr@10": 0.71,
                   "n": prose_n + 12},
    }
    run = {
        "timestamp": name.removesuffix(".json"),
        "git_commit": commit,
        "k": 10,
        "n_in_corpus": prose_n + 12,
        "n_negatives": 7,
        "summary": modes,
        "_file": name,
    }
    if citation:
        run["summary_by_kind"] = {
            "prose": {m: {**v, "n": prose_n} for m, v in modes.items()},
            "citation": {m: {**v, "n": 12} for m, v in modes.items()},
        }
    return run


class TestDiff:
    def test_diffs_every_group_mode_metric(self):
        deltas = diff_runs(make_run("b.json", 0.85), make_run("a.json", 0.90))
        groups = {d.group for d in deltas}
        assert groups == {ALL, "prose", "citation"}
        d = next(x for x in deltas if (x.group, x.mode, x.metric) == (ALL, "hybrid", "recall@10"))
        assert d.change == pytest.approx(-0.05)

    def test_reference_missing_a_kind_yields_no_before(self):
        """Old result files predate summary_by_kind; their kinds diff as new."""
        deltas = diff_runs(make_run("b.json"), make_run("a.json", citation=False))
        d = next(x for x in deltas if (x.group, x.mode) == ("citation", "hybrid"))
        assert d.before is None and d.change is None
        assert not d.is_regression(None)


class TestThreshold:
    def test_adaptive_allows_one_question_fails_two(self):
        """Default threshold is 1.5/n: one flipped question is noise, two fail."""
        one = Delta(group="prose", mode="hybrid", metric="recall@10",
                    before=0.90, after=0.90 - 1 / 42, n=42)
        two = Delta(group="prose", mode="hybrid", metric="recall@10",
                    before=0.90, after=0.90 - 2 / 42, n=42)
        assert one.is_drop and not one.is_regression(None)
        assert two.is_regression(None)

    def test_adaptive_scales_with_group_size(self):
        """The same absolute drop is noise on 12 questions, regression on 42."""
        drop = 0.06
        small = Delta(group="citation", mode="bm25", metric="recall@5",
                      before=1.0, after=1.0 - drop, n=12)
        large = Delta(group="prose", mode="bm25", metric="recall@5",
                      before=1.0, after=1.0 - drop, n=42)
        assert not small.is_regression(None)
        assert large.is_regression(None)

    def test_flat_threshold_overrides_adaptive(self):
        d = Delta(group="citation", mode="hybrid", metric="mrr@10",
                  before=0.83, after=0.80, n=12)
        assert not d.is_regression(None)  # adaptive: 1.5/12 = 0.125 allowed
        assert d.is_regression(0.02)  # flat 0.02: 0.03 drop fails

    def test_gains_never_flag(self):
        d = Delta(group=ALL, mode="vector", metric="recall@5", before=0.5, after=0.9, n=54)
        assert not d.is_drop and not d.is_regression(0.0)


class TestLoadAndPick:
    @pytest.fixture
    def results_dir(self, tmp_path):
        for name, r10 in (("20250101-000000", 0.80), ("20250102-000000", 0.90),
                          ("20250103-000000", 0.85)):
            run = make_run(f"{name}.json", r10)
            run.pop("_file")
            (tmp_path / f"{name}.json").write_text(json.dumps(run), encoding="utf-8")
        (tmp_path / ".gitkeep").touch()
        (tmp_path / "junk.json").write_text("not json", encoding="utf-8")
        return tmp_path

    def test_loads_sorted_and_skips_junk(self, results_dir):
        runs = load_runs(results_dir)
        assert [r["_file"] for r in runs] == [
            "20250101-000000.json", "20250102-000000.json", "20250103-000000.json"]

    def test_pick_previous_and_named_baseline(self, results_dir):
        runs = load_runs(results_dir)
        assert pick_reference(runs, "previous")["_file"] == "20250102-000000.json"
        assert pick_reference(runs, "20250101-000000.json")["_file"] == "20250101-000000.json"
        with pytest.raises(FileNotFoundError):
            pick_reference(runs, "nope.json")

    def test_single_run_has_no_previous(self, results_dir):
        runs = load_runs(results_dir)[:1]
        assert pick_reference(runs, "previous") is None

    def test_cli_fails_on_regression_and_no_fail_downgrades(self, results_dir, tmp_path, capsys):
        report = tmp_path / "report.html"
        args = ["--results-dir", str(results_dir), "--report", str(report)]
        # 0.90 -> 0.85 is a 2.1-question drop over n=54: regression.
        assert compare.main(args) == 1
        assert compare.main([*args, "--no-fail"]) == 0
        # a flat threshold big enough to absorb the drop passes
        assert compare.main([*args, "--threshold", "0.10"]) == 0
        assert report.exists()

    def test_report_contains_history_commits_and_verdict(self, results_dir, tmp_path):
        report = tmp_path / "report.html"
        compare.main(["--results-dir", str(results_dir), "--report", str(report), "--no-fail"])
        page = report.read_text(encoding="utf-8")
        assert "20250103-000000.json" in page and "20250101-000000.json" in page
        assert "abc1234" in page  # git commit in history
        assert "regressed" in page  # verdict line
        assert "<svg" in page  # trend sparkline
        for kind in (ALL, "prose", "citation"):
            assert kind in page

    def test_write_report_with_no_runs(self, tmp_path):
        assert compare.write_report(results_dir=tmp_path, out=tmp_path / "r.html") is None
