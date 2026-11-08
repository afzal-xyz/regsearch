"""The parameter sandbox: instant slider feedback, and no way to adopt anything.

Latency
-------
Scoring a parameter set is arithmetic over candidates that were fetched once
and cross-encoder scores that were computed once — see
:mod:`retrieval.prescore`, which freezes both to disk. This module loads that
file and calls the tuner's own :func:`retrieval.tune.score`. No embedder, no
reranker, no database: a slider move is a few hundred dict lookups and a sort
per question, so the whole gold set re-ranks in milliseconds.

The one thing that could break that is a parameter set whose fusion surfaces a
chunk the cross-encoder never saw, which would silently become a model call.
It cannot happen here: the sliders are restricted to the axes of
:data:`retrieval.tune.GRID`, which is exactly what the cache was warmed over,
every request checks :meth:`PrescoreCache.covers` first, and the reranker
handed to the scoring code is a stub that raises rather than computes. Three
independent guards, because the failure mode is an eight-second stall that
looks like a hang.

Overfitting
-----------
A slider over a test set is a machine for fooling yourself: evaluate enough
parameter sets and the best one looks good by chance alone, even if every one
of them is worthless. The sandbox does not pretend otherwise.

* Every number is labelled in-sample, because it is.
* The live number is computed on an **explore split**; one deterministic fold
  is held back and is not touched until you ask for it, and asking is counted.
  A holdout you consult after every slider move is just another training set.
* The delta against production is judged against the tuner's own noise margin
  of 1/n. Inside it, the honest reading is "no evidence".
* The number of parameter sets you have evaluated is displayed, because the
  best of fifty draws beats the mean by chance and the only defence is knowing
  how many draws you took.

Governance
----------
There is no write path to ``retrieval/params.json`` from this module. It
imports :mod:`retrieval.params` to *read* the current values and never calls
``save``. Adopting means running ``retrieval.tune`` — which cross-validates,
applies the margin, and records provenance — and the sandbox's only
adoption-shaped control is a button that starts exactly that subprocess.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from eval import run as eval_run
from retrieval import params as params_mod
from retrieval import prescore
from retrieval import tune as tune_mod
from retrieval.params import FusionParams

logger = logging.getLogger(__name__)

# The fold held back from the live number. 5 folds -> a 20% holdout, matching
# the tuner's default so the two are talking about comparable splits.
HOLDOUT_FOLDS = tune_mod.DEFAULT_FOLDS
HOLDOUT_FOLD = 0

HYBRID, RERANK = "hybrid", "rerank"


class NotWarmed(RuntimeError):
    """A parameter set outside the warmed region, or no cache at all."""


class _RefuseToScore:
    """A reranker that cannot rerank.

    Passed into the tuner's scoring path as a backstop. If a code change ever
    lets an unwarmed parameter set through the coverage check, this turns a
    silent eight-second model call inside a web request into a loud error.
    """

    model_name = "<sandbox: scoring is cache-only>"

    def score(self, query, passages):  # pragma: no cover - the guard that must not fire
        raise NotWarmed(
            f"the sandbox tried to cross-encoder-score {len(passages)} passage(s) for "
            f"{query[:40]!r}; the prescore cache does not cover this parameter set"
        )


_REFUSE = _RefuseToScore()


# --- the slider axes --------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    name: str
    label: str
    values: tuple  # the allowed positions, in order
    help: str

    def index_of(self, value) -> int:
        return self.values.index(value) if value in self.values else 0

    @property
    def keys(self) -> tuple[str, ...]:
        """Form values for each position — what the sliders actually submit."""
        return tuple(_axis_key(v) for v in self.values)


def _axes() -> list[Axis]:
    """Sliders, derived from :data:`retrieval.tune.GRID` so they cannot drift.

    The sliders are discrete and span exactly the tuner's search space. That
    is a deliberate restriction rather than a shortcut: a point you can reach
    here is a point the tuner will also consider, so "the sandbox liked this"
    is a claim ``retrieval.tune`` can check. Exploring outside the grid means
    widening ``GRID`` in the tuner — a code change that goes through review,
    which is the right amount of friction for changing the search space.
    """
    grid = tune_mod.GRID
    bands = grid["bm25_contrast_band"]
    return [
        Axis("rrf_k", "RRF k", grid["rrf_k"],
             "Smaller k lets a retriever's own top pick outweigh agreement between the two."),
        Axis("bm25_weight", "BM25 weight", grid["bm25_weight"],
             "How much lexical evidence counts against semantic, before confidence scaling."),
        Axis("bm25_contrast_anchor", "Contrast anchor", grid["bm25_contrast_anchor"],
             "Which BM25 rank the top score is compared against to judge confidence."),
        Axis("bm25_contrast_band", "Confidence band (lo, hi)", bands,
             "Contrast below lo means BM25 is ignored; above hi it is trusted outright."),
        Axis("agreement_scale", "Agreement scale", grid["agreement_scale"],
             "Damps BM25 when its top sections do not overlap vector's. 0 disables the gate."),
        Axis("rerank_candidates", "Rerank candidates", grid["rerank_candidates"],
             "How many fused sections the cross-encoder reorders. Rerank mode only."),
    ]


AXES = _axes()
AXES_BY_NAME = {a.name: a for a in AXES}


def params_from(values: dict, base: FusionParams) -> FusionParams:
    """Build a :class:`FusionParams` from slider positions, rejecting off-grid values.

    Values arrive from a form, so each is checked against its axis rather than
    coerced. An off-grid value is refused rather than snapped: snapping would
    show the author a number computed for settings they did not choose.
    """
    changes: dict = {}
    for name, raw in values.items():
        axis = AXES_BY_NAME.get(name)
        if axis is None:
            raise NotWarmed(f"{name!r} is not a sandbox parameter")
        picked = _coerce(axis, raw)
        if name == "bm25_contrast_band":
            changes["bm25_contrast_lo"], changes["bm25_contrast_hi"] = picked
        else:
            changes[name] = picked
    return base.with_(**changes)


def _coerce(axis: Axis, raw):
    """Match a form value to one of the axis's positions.

    Matching is on the rendered string, so ``"0.5"`` finds ``0.5`` and
    ``"(1.4, 2.2)"`` finds the band tuple, without eval or float equality games.
    """
    if raw in axis.values:
        return raw
    text = str(raw).strip()
    for value in axis.values:
        if text == str(value) or text == _axis_key(value):
            return value
    raise NotWarmed(
        f"{text!r} is not a position on {axis.name}; the sandbox only explores the "
        f"tuner's grid ({', '.join(_axis_key(v) for v in axis.values)})"
    )


def _axis_key(value) -> str:
    """Stable form value for an axis position."""
    return ",".join(str(v) for v in value) if isinstance(value, tuple) else str(value)


def axis_value_of(axis: Axis, p: FusionParams):
    if axis.name == "bm25_contrast_band":
        return (p.bm25_contrast_lo, p.bm25_contrast_hi)
    return getattr(p, axis.name)


# --- metrics ----------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    objective: float
    recall5: float
    recall10: float
    mrr10: float
    n: int
    by_kind: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def of(cls, score: tune_mod.Score) -> Metrics:
        """Flatten the tuner's per-kind score into overall means.

        The kinds partition the questions, so a count-weighted mean of the
        per-kind means is the exact overall mean — no re-scoring needed.
        """
        rows = list(score.by_kind.values())
        n = sum(r["n"] for r in rows) or 0

        def mean(metric: str) -> float:
            return sum(r[metric] * r["n"] for r in rows) / n if n else 0.0

        return cls(
            objective=score.objective,
            recall5=mean("recall@5"),
            recall10=mean("recall@10"),
            mrr10=mean("mrr@10"),
            n=n,
            by_kind=score.by_kind,
        )


@dataclass(frozen=True)
class Movement:
    question: str
    kind: str
    before: int | None
    after: int | None

    @property
    def direction(self) -> str:
        if self.before == self.after:
            return "same"
        if self.before is None:
            return "fixed"
        if self.after is None:
            return "broken"
        return "better" if self.after < self.before else "worse"


WITHIN_NOISE, BETTER, WORSE = "within-noise", "better-in-sample", "worse"


@dataclass(frozen=True)
class Evaluation:
    params: FusionParams
    baseline_params: FusionParams
    mode: str
    split: str
    metrics: Metrics
    baseline: Metrics
    margin: float
    probes: int
    moved: tuple[Movement, ...] = ()

    @property
    def delta(self) -> float:
        return self.metrics.objective - self.baseline.objective

    @property
    def verdict(self) -> str:
        if self.delta >= self.margin:
            return BETTER
        if self.delta <= -self.margin:
            return WORSE
        return WITHIN_NOISE

    @property
    def inert(self) -> bool:
        """Changed parameters, identical metrics — and why that is not a bug.

        recall and MRR only see where the *correct* section landed. A fusion
        change reshuffles everything, but if it moves no gold answer across a
        rank boundary the metrics are byte-identical. On a small or easy
        corpus that is the common case, and it looks exactly like a broken
        slider, so say so rather than letting someone debug a non-problem.
        """
        return self.params != self.baseline_params and not self.moved

    @property
    def reading(self) -> str:
        """What the delta does and does not license, in words."""
        if self.verdict == WITHIN_NOISE:
            return (
                f"Inside the noise margin ({self.margin:.4f} = 1/n). One question flipping "
                f"moves the objective this far, so this is no evidence of anything."
            )
        if self.verdict == WORSE:
            return f"Worse than production by more than one question's worth ({self.margin:.4f})."
        return (
            f"Better in-sample by more than 1/n — on the {self.metrics.n} questions you are "
            f"looking at. That is the number you have been steering toward, so it is not "
            f"evidence it generalises. Run the tuner: it chooses on one fold and scores on "
            f"another, and adopts only if the held-out gain survives."
        )


# --- the sandbox ------------------------------------------------------------


def _splits(questions: list[tune_mod.Question]) -> tuple[list[int], list[int]]:
    """(explore, holdout) question indices — deterministic and stratified by kind."""
    folds = tune_mod.folds(questions, HOLDOUT_FOLDS)
    holdout = sorted(folds[HOLDOUT_FOLD])
    held = set(holdout)
    return [i for i in range(len(questions)) if i not in held], holdout


class Sandbox:
    """A loaded prescore cache plus the bookkeeping that keeps users honest."""

    def __init__(self, cache_path: Path | str = prescore.DEFAULT_CACHE):
        self.cache_path = Path(cache_path)
        self._cache: prescore.PrescoreCache | None = None
        self._mtime: int | None = None
        self._explore: list[int] = []
        self._holdout: list[int] = []
        self._lock = threading.Lock()
        self.probes: set[tuple] = set()
        self.holdout_peeks = 0

    # -- loading --

    @property
    def loaded(self) -> bool:
        return self._cache is not None

    def cache(self) -> prescore.PrescoreCache:
        """The loaded cache, re-read when the file on disk has changed.

        The warm job runs in another process, so the file can be replaced
        underneath a long-lived server. Keying on mtime means finishing a
        re-warm is enough — nobody has to remember to restart anything.
        """
        with self._lock:
            mtime = self.cache_path.stat().st_mtime_ns if self.cache_path.exists() else None
            if self._cache is None or mtime != self._mtime:
                self._cache = prescore.load(self.cache_path)
                self._mtime = mtime
                self._explore, self._holdout = _splits(self._cache.questions)
            return self._cache

    def reload(self) -> None:
        """Forget the loaded cache (after a re-warm) but keep the probe count.

        The count deliberately survives: the searching you have already done
        still happened, and a fresh cache does not un-bias it.
        """
        with self._lock:
            self._cache = None

    def questions(self, split: str) -> list[tune_mod.Question]:
        cache = self.cache()
        idx = self._holdout if split == "holdout" else self._explore
        return [cache.questions[i] for i in idx]

    def staleness(self, specs, gold, rerank_model: str | None) -> list[str]:
        """Why the cache no longer describes the live setup (empty = fresh).

        Reuses the tuner's rule rather than restating it: a cache warmed on a
        different corpus or gold set is not merely out of date, it is scoring
        questions that are no longer the benchmark.
        """
        cache = self.cache()
        model = rerank_model if cache.reranked else None
        return tune_mod.staleness(cache.provenance, specs, gold, model)

    # -- scoring --

    def evaluate(
        self,
        p: FusionParams,
        *,
        mode: str = RERANK,
        split: str = "explore",
        baseline: FusionParams | None = None,
        with_movements: bool = True,
    ) -> Evaluation:
        """Score ``p`` against the baseline on one split. Milliseconds, no models."""
        cache = self.cache()
        rerank = mode == RERANK
        if rerank:
            if not cache.reranked:
                raise NotWarmed(
                    "this cache was warmed with --no-rerank, so it holds no cross-encoder "
                    "scores; re-warm without that flag to use rerank mode"
                )
            for candidate in (p, baseline or cache.base_params):
                if not cache.covers(candidate):
                    raise NotWarmed(
                        "that parameter set is outside the warmed region — its fusion would "
                        "surface chunks the cross-encoder never scored. Re-warm, or switch "
                        "to hybrid mode, which needs no cross-encoder at all."
                    )

        base = baseline if baseline is not None else params_mod.current()
        qs = self.questions(split)
        kw = {"rerank": rerank, "conn": None, "reranker": _REFUSE}

        metrics = Metrics.of(tune_mod.score(qs, p, **kw))
        baseline_metrics = Metrics.of(tune_mod.score(qs, base, **kw))

        with self._lock:
            self.probes.add((tune_mod._fusion_key(p), p.rerank_candidates, mode))
            probes = len(self.probes)

        moved: tuple[Movement, ...] = ()
        if with_movements and p != base:
            moved = movements(qs, p, base, **kw)

        return Evaluation(
            params=p,
            baseline_params=base,
            mode=mode,
            split=split,
            metrics=metrics,
            baseline=baseline_metrics,
            margin=(1.0 / len(qs)) if qs else 0.0,
            probes=probes,
            moved=moved,
        )

    def peek_holdout(self, p: FusionParams, *, mode: str = RERANK) -> Evaluation:
        """Score on the held-back fold, and count that it was looked at."""
        with self._lock:
            self.holdout_peeks += 1
        return self.evaluate(p, mode=mode, split="holdout", with_movements=False)


def movements(
    qs: Sequence[tune_mod.Question], p: FusionParams, base: FusionParams, **kw
) -> tuple[Movement, ...]:
    """Per-question rank change between two parameter sets, worst news first."""
    order = {"broken": 0, "worse": 1, "fixed": 2, "better": 3, "same": 4}
    out = []
    for q in qs:
        before = eval_run.first_hit_rank(tune_mod.ranked_sections(q, base, k=10, **kw), q.relevant)
        after = eval_run.first_hit_rank(tune_mod.ranked_sections(q, p, k=10, **kw), q.relevant)
        if before == after:
            continue
        out.append(Movement(q.text, q.kind, before, after))
    return tuple(sorted(out, key=lambda m: (order[m.direction], -abs((m.after or 99)
                                                                     - (m.before or 99)))))
