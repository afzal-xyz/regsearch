"""Read-only view over ``eval/results/*.json`` for the dashboard.

This module adds no metrics and no judgement of its own. Diffing, the
regression threshold and the comparability rule all come from
:mod:`eval.compare`, which is what CI runs — if the dashboard and the gate
ever disagree about whether something regressed, one of them is lying, so
there is only one implementation.

What it does add is the per-question view the CLI has no room for: a matrix of
where each gold question's answer ranked, in each mode, and which questions
moved between two runs. That is the diagnostic that says *why* a mean moved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from eval import compare
from eval.compare import ALL, METRICS, Delta

DEFAULT_RESULTS_DIR = compare.DEFAULT_RESULTS_DIR
RESULT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$")

# Cell states in the status matrix.
TOP5 = "top5"  # answer in the first 5 — good
TOP10 = "top10"  # found, but ranked 6..10 — recall@10 only
MISS = "miss"  # not retrieved at all
NEGATIVE = "negative"  # no correct answer exists; rank is undefined, not a miss


# --- run metadata -----------------------------------------------------------


@dataclass(frozen=True)
class RunMeta:
    """The header of a run: everything but the per-question detail."""

    file: str
    timestamp: str
    git_commit: str | None
    modes: tuple[str, ...]
    k: int | None
    n_in_corpus: int | None
    n_negatives: int | None
    gold_n: int | None
    gold_sha256: str | None
    rerank_model: str | None
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.timestamp} ({self.git_commit or 'no commit'})"


def meta_of(run: dict) -> RunMeta:
    fp = run.get("gold_fingerprint") or {}
    return RunMeta(
        file=run.get("_file", ""),
        timestamp=str(run.get("timestamp", run.get("_file", ""))),
        git_commit=run.get("git_commit"),
        modes=tuple(run.get("modes") or sorted(run.get("summary", {}))),
        k=run.get("k"),
        n_in_corpus=run.get("n_in_corpus"),
        n_negatives=run.get("n_negatives"),
        gold_n=fp.get("n", run.get("n_in_corpus")),
        gold_sha256=fp.get("sha256"),
        rerank_model=run.get("rerank_model"),
        params=run.get("params") or {},
    )


def load_runs(results_dir: Path | str = DEFAULT_RESULTS_DIR) -> list[dict]:
    """Every run, oldest first. Thin pass-through so callers import one module."""
    return compare.load_runs(results_dir)


def list_runs(results_dir: Path | str = DEFAULT_RESULTS_DIR) -> list[RunMeta]:
    """Run headers, newest first (the order a dashboard reads in)."""
    return [meta_of(r) for r in reversed(load_runs(results_dir))]


def load_run(name: str, results_dir: Path | str = DEFAULT_RESULTS_DIR) -> dict:
    """One run by filename.

    ``name`` is matched against a strict pattern and the resolved path is
    checked to be a direct child of ``results_dir``: it arrives from a URL,
    and ``../../etc/passwd`` is not a run.
    """
    if not RESULT_NAME.match(name):
        raise ValueError(f"{name!r} is not a result filename")
    root = Path(results_dir).resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise FileNotFoundError(f"no eval run named {name!r}")
    for run in load_runs(results_dir):
        if run["_file"] == name:
            return run
    raise FileNotFoundError(f"{name!r} is not a readable eval run")


# --- comparison -------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """``eval.compare``'s verdict, as data rather than printed text."""

    current: dict
    reference: dict | None
    deltas: list[Delta]
    regressions: list[Delta]
    comparable: bool
    note: str
    threshold: float | None

    @property
    def groups(self) -> list[str]:
        """ALL first, then kinds — the order the CLI prints them in."""
        return sorted({d.group for d in self.deltas}, key=lambda g: (g != ALL, g))

    def rows(self, group: str) -> list[Delta]:
        return [d for d in self.deltas if d.group == group]


def compare_runs(
    current: dict, reference: dict | None, *, threshold: float | None = None
) -> Comparison:
    """Diff two runs, suppressing the gate exactly as ``eval.compare`` does.

    When the gold set changed between the runs, the deltas compare means over
    different populations. They are still worth showing — the new absolute
    numbers are the point — but they are not evidence of a regression, so the
    regression list is emptied rather than the deltas hidden.
    """
    if reference is None:
        return Comparison(current, None, [], [], True, "no earlier run to compare against", None)

    deltas = compare.diff_runs(current, reference)
    ok = compare.comparable(current, reference)
    regressions = [d for d in deltas if d.is_regression(threshold)] if ok else []
    note = ""
    if not ok:
        note = (
            f"{compare.describe_gold_change(current, reference)} between these runs, so the "
            f"deltas compare means over different question sets. Shown for information; the "
            f"gate is not firing on them."
        )
    elif not (current.get("gold_fingerprint") and reference.get("gold_fingerprint")):
        note = (
            "one of these runs predates gold-set fingerprinting, so it cannot be confirmed "
            "that both were scored against the same questions."
        )
    return Comparison(current, reference, deltas, regressions, ok, note, threshold)


def latest_comparison(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    against: str = "previous",
    threshold: float | None = None,
) -> Comparison | None:
    """The newest run against ``previous`` or a named baseline. None if no runs."""
    runs = load_runs(results_dir)
    if not runs:
        return None
    return compare_runs(runs[-1], compare.pick_reference(runs, against), threshold=threshold)


# --- per-question diagnostics -----------------------------------------------


@dataclass(frozen=True)
class QuestionRow:
    question: str
    kind: str
    citations: tuple[str, ...]
    ranks: dict[str, int | None]  # mode -> 1-based rank of the first hit
    states: dict[str, str]
    top: dict[str, tuple[str, ...]]  # mode -> the sections it actually returned

    @property
    def negative(self) -> bool:
        return not self.citations

    def worst_rank(self, modes: tuple[str, ...]) -> tuple[int, int, int]:
        """Sort key, ascending = worst first.

        Questions some mode failed to retrieve at all come first, the ones no
        mode found ahead of the ones only one did; then questions every mode
        found, deepest rank first. Negatives sink to the bottom: they have no
        rank, so there is nothing to be worried about.
        """
        if self.negative:
            return (2, 0, 0)
        ranks = [self.ranks.get(m) for m in modes]
        found = [r for r in ranks if r is not None]
        deepest = -max(found) if found else 0
        bucket = 0 if len(found) < len(ranks) else 1
        return (bucket, len(found), deepest)


def _state(rank: int | None, negative: bool) -> str:
    if negative:
        # A negative has no relevant citation, so first_hit_rank is None by
        # construction, for every mode and every system. Scoring it as a
        # "miss" would be reading a guarantee as a result. Whether the system
        # correctly declined to answer is not visible in a retrieval run at
        # all — that is the answer layer's gate, and it is not built yet.
        return NEGATIVE
    if rank is None:
        return MISS
    return TOP5 if rank <= 5 else TOP10


def status_matrix(
    run: dict,
    *,
    modes: tuple[str, ...] | None = None,
    kind: str | None = None,
    only_problems: bool = False,
) -> list[QuestionRow]:
    """One row per gold question, one column per mode, worst questions first."""
    cols = tuple(modes or run.get("modes") or ())
    rows: list[QuestionRow] = []
    for entry in run.get("details", []):
        entry_kind = entry.get("kind", "prose")
        if kind and entry_kind != kind:
            continue
        citations = tuple(entry.get("relevant_citations") or [])
        per_mode = entry.get("modes", {})
        cols_here = cols or tuple(per_mode)
        ranks = {m: per_mode.get(m, {}).get("first_hit_rank") for m in cols_here}
        states = {m: _state(ranks[m], not citations) for m in cols_here}
        if only_problems and not any(s in (MISS, TOP10) for s in states.values()):
            continue
        rows.append(
            QuestionRow(
                question=entry.get("question", ""),
                kind=entry_kind,
                citations=citations,
                ranks=ranks,
                states=states,
                top={
                    m: tuple(per_mode.get(m, {}).get("top_sections") or []) for m in cols_here
                },
            )
        )
    if cols:
        order = cols
    elif rows:
        order = tuple(rows[0].ranks)
    else:
        order = ()
    return sorted(rows, key=lambda r: r.worst_rank(order))


# How a question moved between two runs, worst news first.
BROKEN = "broken"  # was found, now missed
WORSE = "worse"
BETTER = "better"
FIXED = "fixed"  # was missed, now found
SAME = "same"
_MOVE_ORDER = {BROKEN: 0, WORSE: 1, FIXED: 2, BETTER: 3, SAME: 4}


@dataclass(frozen=True)
class QuestionDelta:
    question: str
    kind: str
    mode: str
    before: int | None
    after: int | None
    movement: str

    @property
    def sort_key(self) -> tuple[int, int]:
        size = abs((self.after or 99) - (self.before or 99))
        return (_MOVE_ORDER[self.movement], -size)


def _movement(before: int | None, after: int | None) -> str:
    if before == after:
        return SAME
    if before is None:
        return FIXED
    if after is None:
        return BROKEN
    return BETTER if after < before else WORSE


def question_deltas(
    current: dict, reference: dict, *, mode: str, include_same: bool = False
) -> list[QuestionDelta]:
    """Per-question rank movement for one mode, matched on question text.

    Questions present in only one of the runs are skipped: the gold set
    changing is the common reason two runs differ, and a question that did not
    exist before has not moved.
    """
    before_by_q = {
        e.get("question"): e.get("modes", {}).get(mode, {}).get("first_hit_rank")
        for e in reference.get("details", [])
    }
    out: list[QuestionDelta] = []
    for entry in current.get("details", []):
        question = entry.get("question")
        if question not in before_by_q:
            continue
        if not entry.get("relevant_citations"):
            continue  # negatives have no rank to move
        after = entry.get("modes", {}).get(mode, {}).get("first_hit_rank")
        before = before_by_q[question]
        movement = _movement(before, after)
        if movement == SAME and not include_same:
            continue
        out.append(
            QuestionDelta(
                question=question,
                kind=entry.get("kind", "prose"),
                mode=mode,
                before=before,
                after=after,
                movement=movement,
            )
        )
    return sorted(out, key=lambda d: d.sort_key)


# --- history ----------------------------------------------------------------


def metric_series(
    runs: list[dict], group: str, mode: str, metric: str
) -> list[float | None]:
    """One metric across every run, oldest first (None where absent)."""
    return compare._metric_series(runs, group, mode, metric)


def sparkline(values: list[float | None]) -> str:
    """Inline SVG trend, on the same 0..1 scale ``eval/report.html`` uses."""
    return compare._sparkline(values)


def summary_table(run: dict, group: str = ALL) -> dict[str, dict]:
    """``{mode: {metric: value, "n": int}}`` for one group of one run."""
    return compare.summaries_by_group(run).get(group, {})


def groups_of(run: dict) -> list[str]:
    return sorted(compare.summaries_by_group(run), key=lambda g: (g != ALL, g))


__all__ = [
    "ALL",
    "METRICS",
    "Comparison",
    "QuestionDelta",
    "QuestionRow",
    "RunMeta",
    "compare_runs",
    "groups_of",
    "latest_comparison",
    "list_runs",
    "load_run",
    "load_runs",
    "metric_series",
    "meta_of",
    "question_deltas",
    "sparkline",
    "status_matrix",
    "summary_table",
]
