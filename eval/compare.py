"""Compare eval runs, enforce a regression threshold, and render a report.

``eval/run.py`` leaves one JSON per run in ``eval/results/``; this module turns
that pile into answers:

- **Diff** the latest run against the previous one (default) or against a
  pinned baseline file, per mode (vector / bm25 / hybrid) and per question
  kind (prose / citation / ...), for every metric.
- **Gate**: any metric drop is printed as a warning; a drop bigger than the
  threshold fails the run (exit 1), so CI can block a retrieval regression.
- **History**: every run, oldest first, with the git commit it ran on —
  rendered as tables and trend sparklines in a self-contained
  ``eval/report.html``.

Threshold
---------
The default is *adaptive*: a group's metrics move in steps of roughly one
question (1/n), so on 42 prose questions a single flipped question moves
recall by 0.024 while on 12 citation questions it moves it by 0.083. A fixed
absolute threshold either ignores real two-question losses in the big group or
fires on one-question noise in the small one. The default allows a drop of
``1.5 / n`` — one question's worth of movement (hand-written gold questions
flip on borderline rank changes; that is noise, not regression) but never
two — scaled to each group's own size. ``--threshold 0.05`` switches to a flat
absolute limit when you want one number everywhere.

CLI:
    python -m eval.compare                       # latest vs previous
    python -m eval.compare --against 20261024-192651.json   # vs a baseline
    python -m eval.compare --threshold 0.05 --no-fail
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RESULTS_DIR = Path("eval/results")
DEFAULT_REPORT = Path("eval/report.html")
METRICS = ("recall@5", "recall@10", "mrr@10")
# "ALL" aggregates every in-corpus question; kinds break it down.
ALL = "ALL"
# Adaptive threshold: allow one flipped question (1/n) per group, fail at two.
# 1.5 sits between, so a single borderline question is noise and a second one
# is a regression, at every group size.
QUESTIONS_OF_SLACK = 1.5


# --- loading ----------------------------------------------------------------


def load_runs(results_dir: Path | str = DEFAULT_RESULTS_DIR) -> list[dict]:
    """All eval runs in ``results_dir``, oldest first (filenames are stamps)."""
    runs = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # a half-written or foreign file is not a run
        if "summary" not in run:
            continue
        run["_file"] = path.name
        runs.append(run)
    return runs


def summaries_by_group(run: dict) -> dict[str, dict]:
    """``{group: {mode: {metric: value, "n": int}}}`` with ALL first.

    Older result files predate ``summary_by_kind``; they contribute only ALL.
    """
    groups = {ALL: run["summary"]}
    groups.update(run.get("summary_by_kind", {}))
    return groups


# --- diffing ----------------------------------------------------------------


@dataclass
class Delta:
    group: str  # ALL or a question kind
    mode: str
    metric: str
    before: float | None  # None when the reference lacks this group/mode
    after: float
    n: int  # question count in the current run's group

    @property
    def change(self) -> float | None:
        return None if self.before is None else self.after - self.before

    def allowed_drop(self, threshold: float | None) -> float:
        if threshold is not None:
            return threshold
        return QUESTIONS_OF_SLACK / self.n if self.n else 0.0

    def is_regression(self, threshold: float | None) -> bool:
        return self.change is not None and -self.change > self.allowed_drop(threshold)

    @property
    def is_drop(self) -> bool:
        return self.change is not None and self.change < 0


def diff_runs(current: dict, reference: dict) -> list[Delta]:
    """Every (group, mode, metric) delta between two runs, current's groups."""
    ref_groups = summaries_by_group(reference)
    deltas = []
    for group, modes in summaries_by_group(current).items():
        ref_modes = ref_groups.get(group, {})
        for mode, values in modes.items():
            ref_values = ref_modes.get(mode, {})
            for metric in METRICS:
                after = values.get(metric)
                if after is None:
                    continue
                deltas.append(
                    Delta(
                        group=group,
                        mode=mode,
                        metric=metric,
                        before=ref_values.get(metric),
                        after=after,
                        n=values.get("n", 0),
                    )
                )
    return deltas


def pick_reference(runs: list[dict], against: str) -> dict | None:
    """The run to compare the latest against: ``previous`` or a filename."""
    if against == "previous":
        return runs[-2] if len(runs) >= 2 else None
    name = Path(against).name
    for run in runs:
        if run["_file"] == name:
            return run
    raise FileNotFoundError(f"no eval run named {name!r} in the results directory")


# --- console report ---------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def print_comparison(deltas: list[Delta], threshold: float | None) -> list[Delta]:
    """Print the per-group tables; return the deltas that breach the threshold."""
    regressions = []
    groups = sorted({d.group for d in deltas}, key=lambda g: (g != ALL, g))
    for group in groups:
        rows = [d for d in deltas if d.group == group]
        n = rows[0].n if rows else 0
        print(f"\n{group} (n = {n})")
        header = f"{'mode':<8} {'metric':<10} {'before':>7} {'after':>7} {'change':>8}"
        print(header)
        print("-" * len(header))
        for d in rows:
            change = d.change
            mark = ""
            if d.is_regression(threshold):
                mark = "  ✗ regression"
                regressions.append(d)
            elif d.is_drop:
                mark = "  ⚠ drop (within threshold)"
            change_s = "      —" if change is None else f"{change:+8.3f}"
            print(f"{d.mode:<8} {d.metric:<10} {_fmt(d.before):>7} {d.after:7.3f} {change_s}{mark}")
    return regressions


# --- HTML report ------------------------------------------------------------


def _sparkline(values: list[float | None], width: int = 220, height: int = 36) -> str:
    """Inline SVG of ``values`` on a fixed 0..1 scale (metrics are fractions)."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if not pts:
        return ""
    step = width / max(len(values) - 1, 1)
    coords = [(round(i * step, 1), round(height - v * (height - 4) - 2, 1)) for i, v in pts]
    line = " ".join(f"{x},{y}" for x, y in coords)
    last = coords[-1]
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline fill="none" stroke="#2563eb" stroke-width="1.5" points="{line}"/>'
        f'<circle cx="{last[0]}" cy="{last[1]}" r="2.5" fill="#2563eb"/></svg>'
    )


def _metric_series(runs: list[dict], group: str, mode: str, metric: str) -> list[float | None]:
    return [summaries_by_group(r).get(group, {}).get(mode, {}).get(metric) for r in runs]


_CSS = """
body { font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 70rem;
       color: #1f2937; padding: 0 1rem; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; margin: .5rem 0 1.5rem; }
th, td { padding: .3rem .8rem; border-bottom: 1px solid #e5e7eb; text-align: right; }
th { background: #f3f4f6; } td:first-child, th:first-child { text-align: left; }
.regression { background: #fee2e2; font-weight: 600; }
.drop { background: #fef9c3; }
.gain { color: #15803d; }
.muted { color: #9ca3af; }
code { background: #f3f4f6; padding: 0 .3em; border-radius: 3px; }
"""


def _delta_table(deltas: list[Delta], group: str, threshold: float | None) -> str:
    rows = []
    for d in (x for x in deltas if x.group == group):
        change = d.change
        cls, change_s = "", "—"
        if change is not None:
            change_s = f"{change:+.3f}"
            if d.is_regression(threshold):
                cls = "regression"
            elif change < 0:
                cls = "drop"
            elif change > 0:
                cls = "gain"
        before = "—" if d.before is None else f"{d.before:.3f}"
        rows.append(
            f'<tr class="{cls}"><td>{d.mode}</td><td>{d.metric}</td>'
            f"<td>{before}</td><td>{d.after:.3f}</td><td>{change_s}</td>"
            f"<td>{d.allowed_drop(threshold):.3f}</td></tr>"
        )
    n = next((d.n for d in deltas if d.group == group), 0)
    return (
        f"<h2>{html.escape(group)} <span class='muted'>(n = {n})</span></h2>"
        "<table><tr><th>mode</th><th>metric</th><th>before</th><th>after</th>"
        "<th>change</th><th>allowed drop</th></tr>" + "".join(rows) + "</table>"
    )


def _history_section(runs: list[dict]) -> str:
    recent = runs[-30:]  # the page stays readable; the JSONs keep everything
    head = (
        "<tr><th>run</th><th>commit</th><th>questions</th>"
        + "".join(f"<th>{m} r@10</th><th>{m} MRR</th>" for m in ("vector", "bm25", "hybrid"))
        + "</tr>"
    )
    body = []
    for r in reversed(recent):  # newest first in the table
        cells = [
            f"<td>{html.escape(r['_file'])}</td>",
            f"<td><code>{html.escape(r.get('git_commit') or '?')}</code></td>",
            f"<td>{r.get('n_in_corpus', '?')} + {r.get('n_negatives', '?')} neg</td>",
        ]
        for mode in ("vector", "bm25", "hybrid"):
            s = r["summary"].get(mode, {})
            for metric in ("recall@10", "mrr@10"):
                v = s.get(metric)
                cells.append(f"<td>{'—' if v is None else f'{v:.3f}'}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    kinds = sorted({k for r in recent for k in summaries_by_group(r) if k != ALL})
    charts = []
    for group in [ALL, *kinds]:
        row = f"<tr><td>{html.escape(group)}</td>"
        for mode in ("vector", "bm25", "hybrid"):
            row += f"<td>{_sparkline(_metric_series(recent, group, mode, 'recall@10'))}</td>"
        charts.append(row + "</tr>")
    return (
        "<h2>History</h2>"
        f"<p>Last {len(recent)} of {len(runs)} runs, newest first. Trend charts plot "
        "recall@10 on a fixed 0–1 scale, oldest → newest.</p>"
        "<table><tr><th>group</th><th>vector</th><th>bm25</th><th>hybrid</th></tr>"
        + "".join(charts)
        + "</table><table>"
        + head
        + "".join(body)
        + "</table>"
    )


def render_report(
    runs: list[dict],
    deltas: list[Delta],
    reference: dict | None,
    threshold: float | None,
) -> str:
    current = runs[-1]
    regressions = [d for d in deltas if d.is_regression(threshold)]
    if reference is None:
        verdict = '<p class="muted">No earlier run to compare against.</p>'
        comparison = ""
    else:
        status = (
            f'<strong style="color:#b91c1c">✗ {len(regressions)} metric(s) regressed</strong>'
            if regressions
            else '<strong class="gain">✓ no regressions</strong>'
        )
        thr = (
            f"flat {threshold:.3f}"
            if threshold is not None
            else f"adaptive, {QUESTIONS_OF_SLACK:g} questions’ worth (1.5/n per group)"
        )
        verdict = (
            f"<p>Latest <code>{html.escape(current['_file'])}</code> vs "
            f"<code>{html.escape(reference['_file'])}</code> — {status} "
            f"<span class='muted'>(threshold: {thr})</span></p>"
        )
        groups = sorted({d.group for d in deltas}, key=lambda g: (g != ALL, g))
        comparison = "".join(_delta_table(deltas, g, threshold) for g in groups)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>RegSearch eval report</title><style>{_CSS}</style></head><body>"
        "<h1>RegSearch eval report</h1>"
        f"<p class='muted'>Generated from eval/results/ — latest run "
        f"{html.escape(current.get('timestamp', '?'))}, commit "
        f"<code>{html.escape(current.get('git_commit') or '?')}</code>, "
        f"gold set {current.get('n_in_corpus', '?')} in-corpus + "
        f"{current.get('n_negatives', '?')} negatives.</p>"
        + verdict
        + comparison
        + _history_section(runs)
        + "</body></html>"
    )


def write_report(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out: Path | str = DEFAULT_REPORT,
    against: str = "previous",
    threshold: float | None = None,
) -> Path | None:
    """Regenerate the HTML report; returns its path, or None with no runs."""
    runs = load_runs(results_dir)
    if not runs:
        return None
    reference = pick_reference(runs, against)
    deltas = diff_runs(runs[-1], reference) if reference else []
    out = Path(out)
    out.write_text(render_report(runs, deltas, reference, threshold), encoding="utf-8")
    return out


# --- CLI --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.compare", description=__doc__)
    parser.add_argument(
        "--against",
        default="previous",
        help="'previous' (default) or a results filename to use as the baseline",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="flat max allowed drop per metric (default: adaptive 1.5/n per group)",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="report regressions but always exit 0 (warn-only mode)",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    runs = load_runs(args.results_dir)
    if not runs:
        print("no eval runs found — run `python -m eval.run` first", file=sys.stderr)
        return 2
    reference = pick_reference(runs, args.against)

    current = runs[-1]
    print(f"latest run:  {current['_file']}  (commit {current.get('git_commit') or '?'})")
    if reference is None:
        print("no reference run to compare against — history only")
        regressions: list[Delta] = []
        deltas: list[Delta] = []
    else:
        print(f"compared to: {reference['_file']}  (commit {reference.get('git_commit') or '?'})")
        deltas = diff_runs(current, reference)
        regressions = print_comparison(deltas, args.threshold)

    out = Path(args.report)
    out.write_text(render_report(runs, deltas, reference, args.threshold), encoding="utf-8")
    print(f"\nwrote {out}")

    if regressions:
        print(f"\n✗ {len(regressions)} metric(s) dropped beyond the threshold:", file=sys.stderr)
        for d in regressions:
            print(
                f"  {d.group} / {d.mode} / {d.metric}: {d.before:.3f} → {d.after:.3f} "
                f"(allowed drop {d.allowed_drop(args.threshold):.3f})",
                file=sys.stderr,
            )
        return 0 if args.no_fail else 1
    print("✓ no regressions beyond the threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
