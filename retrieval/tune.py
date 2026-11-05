"""Re-tune the fusion and rerank parameters against the current corpus and gold set.

Why this exists
---------------
The hybrid fusion constants are calibrated against a score *distribution* —
how sharply BM25's top hit stands out from its 20th, how many ranks separate a
good hit from a bad one — and that distribution moves every time the corpus
changes. The code said "re-tune when the corpus grows". Nobody did when Title
1 arrived, and hybrid's prose MRR fell below plain vector search with no
stage noticing. This module is that step, made mechanical and run by the
pipeline before every eval, so parameters can no longer silently go stale.

How it avoids fooling itself
----------------------------
Picking the parameters that score best on the gold set and then reporting the
gold-set score is circular: the number will always look good, and it says
nothing about the next title. So:

1. **Cross-validation.** Questions are split into folds. For each fold the
   best parameters are chosen on the *other* folds and scored on this one.
   The average of those held-out scores is an honest estimate of what tuning
   actually buys on questions it has not seen.
2. **A noise margin.** The new parameters are adopted only if that honest
   estimate beats the current parameters by more than one question's worth
   (1/n). A metric over a hundred-odd questions moves in steps of ~0.01, and
   a gain inside one step is a coin flip, not a finding. Otherwise the
   current parameters are kept — and their provenance is updated, because
   "checked against this corpus and still best" is exactly what provenance
   should record.
3. **Stability tie-break.** Among parameter sets within noise of each other
   the one nearest the current values wins, so re-tuning does not wander
   across a flat plateau for no reason.

What is tuned, and what is not
------------------------------
Tuned: the RRF constant, the BM25 weight, the BM25 confidence band (anchor,
lo, hi), the retriever-agreement gate, and how many candidates hybrid hands
to the cross-encoder. All of
these are arithmetic over candidates that are fetched once per question, so
hundreds of combinations cost seconds.

Not tuned: the models. Which bi-encoder embeds the chunks and which
cross-encoder reranks them are choices with a 130 MB download and a full
re-index behind them; swapping them is a deliberate experiment, not a knob
the pipeline should turn on its own. See "Limitations" in the README.

CLI:
    python -m retrieval.tune                # tune, write retrieval/params.json
    python -m retrieval.tune --dry-run      # report only
    python -m retrieval.tune --no-rerank    # fusion only (no cross-encoder)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from eval import run as eval_run
from index.embedder import DEFAULT_MODEL, Embedder, contextualize
from index.store import DEFAULT_DB, connect
from index.tokenizer import ANALYZER_VERSION
from pipeline import manifest
from retrieval import params as params_mod
from retrieval import search
from retrieval.params import FusionParams, Provenance

logger = logging.getLogger(__name__)

DEFAULT_FOLDS = 5
FOLD_SEED = 20260730
POOL = search.POOL
MAX_RERANK = 20  # widest candidate count in the grid; bounds cross-encoder work

# The grid. Each axis spans the range the README's experiments found
# plausible, plus a step beyond on each side so a drift is detectable. rrf_k
# reaches down to 2 because under RRF at k=10 the gap between a retriever's
# rank 1 and rank 2 is 1/11 - 1/12 = 0.0076 — smaller than almost any BM25
# contribution — so a single co-occurring section outvotes vector's own first
# choice; small k is the only lever that lets a retriever's top pick count.
GRID = {
    "rrf_k": (2, 3, 5, 10, 20, 60),
    "bm25_weight": (0.25, 0.5, 0.75, 1.0, 1.25),
    "bm25_contrast_anchor": (10, 20),
    "bm25_contrast_band": ((1.2, 1.8), (1.2, 2.2), (1.4, 1.8), (1.4, 2.2), (1.4, 2.6),
                           (1.6, 2.2), (1.6, 2.6), (1.6, 3.0), (1.8, 2.6), (1.8, 3.0)),
    "rerank_candidates": (10, 12, 15, 20),
    "agreement_scale": (0.0, 0.2, 0.3, 0.5),
}


def grid(base: FusionParams, *, rerank: bool) -> list[FusionParams]:
    out = []
    rerank_axis = GRID["rerank_candidates"] if rerank else (base.rerank_candidates,)
    for rrf_k, w, anchor, (lo, hi), rc, ag in itertools.product(
        GRID["rrf_k"], GRID["bm25_weight"], GRID["bm25_contrast_anchor"],
        GRID["bm25_contrast_band"], rerank_axis, GRID["agreement_scale"],
    ):
        out.append(base.with_(
            rrf_k=rrf_k, bm25_weight=w, bm25_contrast_anchor=anchor,
            bm25_contrast_lo=lo, bm25_contrast_hi=hi, rerank_candidates=rc,
            agreement_scale=ag,
        ))
    if base not in out:
        out.append(base)  # the incumbent is always a candidate
    return out


def distance(a: FusionParams, b: FusionParams) -> float:
    """How far apart two parameter sets are, for the stability tie-break."""
    return (
        abs(a.rrf_k - b.rrf_k) / 60
        + abs(a.bm25_weight - b.bm25_weight)
        + abs(a.bm25_contrast_anchor - b.bm25_contrast_anchor) / 20
        + abs(a.bm25_contrast_lo - b.bm25_contrast_lo)
        + abs(a.bm25_contrast_hi - b.bm25_contrast_hi)
        + abs(a.rerank_candidates - b.rerank_candidates) / 20
        + abs(a.agreement_scale - b.agreement_scale)
    )


# --- per-question cache -----------------------------------------------------


@dataclass
class Question:
    text: str
    relevant: set[str]
    kind: str
    cands: search.Candidates
    chunks: dict[str, search.SearchResult] = field(default_factory=dict)  # chunk_id -> result
    ce: dict[str, float] = field(default_factory=dict)  # chunk_id -> cross-encoder score


def _collect(gold: list[dict], *, conn, embedder) -> list[Question]:
    """Fetch candidates for every in-corpus gold question, once."""
    out = []
    for g in gold:
        if not g["relevant_citations"]:
            continue  # negatives have no rank to score; see README "Limitations"
        cands = search.retrieve_candidates(g["question"], conn=conn, embedder=embedder, pool=POOL)
        q = Question(g["question"], set(g["relevant_citations"]), g.get("kind", "prose"), cands)
        for r in cands.vector_hits + cands.bm25_chunks:
            q.chunks.setdefault(r.chunk_id, r)
        out.append(q)
    return out


def _ensure_ce(q: Question, chunk_ids: list[str], *, conn, reranker) -> None:
    """Cross-encoder-score any of ``chunk_ids`` not yet scored, in one batch."""
    missing = [c for c in dict.fromkeys(chunk_ids) if c not in q.ce]
    if not missing:
        return
    for cid in missing:
        if cid not in q.chunks:  # a `citing` chunk neither retriever surfaced
            q.chunks[cid] = search._result(conn, cid, 0.0, [])
    scores = reranker.score(q.text, [contextualize(q.chunks[c]) for c in missing])
    q.ce.update(zip(missing, (float(s) for s in scores), strict=True))


def _prescore(qs: list[Question], candidates: list[FusionParams], *, conn, reranker) -> None:
    """Cross-encoder-score, per question, every chunk any candidate set could
    hand to the reranker — in one batch, with progress.

    Scoring lazily as each parameter set asks works, but it issues many tiny
    batches per question (a few new chunks each time a fusion shifts the top
    20), and a 12-layer cross-encoder on CPU pays per call, not per pair. The
    union across the grid is bounded — every fusion's top ``MAX_RERANK`` is
    drawn from the two retrievers' pools plus the citation hits — so collect it
    once and score it once. This is also where the tuner spends nearly all of
    its wall-clock with a large reranker, hence the progress line.
    """
    fusion_keys = {_fusion_key(p): p for p in candidates}
    total_pairs = 0
    start = time.monotonic()
    for i, q in enumerate(qs, start=1):
        wanted: dict[str, None] = {}
        for p in fusion_keys.values():
            for t in _fused(q, p, MAX_RERANK):
                if not t[4]:  # pinned results are never scored
                    wanted.setdefault(t[1])
        _ensure_ce(q, list(wanted), conn=conn, reranker=reranker)
        total_pairs += len(wanted)
        if i % 10 == 0 or i == len(qs):
            elapsed = time.monotonic() - start
            print(
                f"  cross-encoder: {i}/{len(qs)} questions, {total_pairs} pairs, "
                f"{elapsed:.0f}s elapsed, ~{elapsed / i * (len(qs) - i):.0f}s left",
                flush=True,
            )


# --- scoring ----------------------------------------------------------------


def _fusion_key(p: FusionParams) -> tuple:
    """The fields :func:`search.fuse` reads — everything but ``rerank_candidates``."""
    return (p.rrf_k, p.bm25_weight, p.bm25_contrast_anchor, p.bm25_contrast_lo,
            p.bm25_contrast_hi, p.agreement_scale, p.agreement_k)


def _fused(q: Question, p: FusionParams, n: int) -> list[tuple]:
    """``fuse`` for ``q`` under ``p``, cached on the question across the
    rerank-candidates axis: fusing to 20 and slicing to 10 is identical to
    fusing to 10, so one fusion serves every candidate count."""
    key = _fusion_key(p)
    cache = q.__dict__.setdefault("_fuse_cache", {})
    if key not in cache:
        cache[key] = search.fuse(q.cands, max(MAX_RERANK, n), params=p, pool=POOL)
    return cache[key][:n]


def ranked_sections(
    q: Question, p: FusionParams, *, k: int, rerank: bool, conn=None, reranker=None
) -> list[str]:
    """Section order the pipeline would return for ``q`` under ``p``."""
    if not rerank:
        return [s for s, *_ in _fused(q, p, k)]
    fused = _fused(q, p, max(p.rerank_candidates, k))
    pinned = [t for t in fused if t[4]]
    rest = [t for t in fused if not t[4]]
    _ensure_ce(q, [t[1] for t in rest], conn=conn, reranker=reranker)
    # Same rule as rag.reranker.rerank: stable sort by score, pinned first.
    rest.sort(key=lambda t: -q.ce[t[1]])
    return [t[0] for t in (pinned + rest)[:k]]


def objective(ranked: list[str], relevant: set[str]) -> float:
    """Per-question score: half recall@10, half MRR@10.

    recall@10 is the ceiling everything downstream (reranker, answer layer)
    can reach; MRR@10 is whether the answer is *near the top*, the metric that
    was stuck. Optimising either alone trades the other away — maximum MRR
    with collapsing recall is the failure the Title 1 numbers showed.
    """
    r10 = eval_run.recall_at_k(ranked, relevant, 10) or 0.0
    mrr = eval_run.reciprocal_rank_at_k(ranked, relevant, 10) or 0.0
    return 0.5 * r10 + 0.5 * mrr


@dataclass
class Score:
    objective: float
    by_kind: dict[str, dict[str, float]]  # kind -> {recall@5, recall@10, mrr@10, n}


def score(qs: list[Question], p: FusionParams, **kw) -> Score:
    total = 0.0
    kinds: dict[str, list[list[float]]] = {}
    for q in qs:
        ranked = ranked_sections(q, p, k=10, **kw)
        total += objective(ranked, q.relevant)
        kinds.setdefault(q.kind, []).append([
            eval_run.recall_at_k(ranked, q.relevant, 5) or 0.0,
            eval_run.recall_at_k(ranked, q.relevant, 10) or 0.0,
            eval_run.reciprocal_rank_at_k(ranked, q.relevant, 10) or 0.0,
        ])
    by_kind = {
        kind: {
            "recall@5": sum(r[0] for r in rows) / len(rows),
            "recall@10": sum(r[1] for r in rows) / len(rows),
            "mrr@10": sum(r[2] for r in rows) / len(rows),
            "n": len(rows),
        }
        for kind, rows in kinds.items()
    }
    return Score(total / len(qs) if qs else 0.0, by_kind)


def per_question(qs: list[Question], p: FusionParams, **kw) -> list[float]:
    return [objective(ranked_sections(q, p, k=10, **kw), q.relevant) for q in qs]


def best_on(
    idx: list[int], table: list[list[float]], candidates: list[FusionParams], current: FusionParams
) -> tuple[FusionParams, float]:
    """Highest mean objective over questions ``idx``; ties (within 1e-9) go to
    the set nearest ``current``. ``table[i][j]`` is candidate i on question j."""
    means = [sum(row[j] for j in idx) / len(idx) for row in table]
    top = max(means)
    tied = [candidates[i] for i, m in enumerate(means) if top - m < 1e-9]
    return min(tied, key=lambda p: distance(p, current)), top


def folds(qs: list[Question], n_folds: int, seed: int = FOLD_SEED) -> list[list[int]]:
    """Stratified by kind so every fold has citation lookups, not just prose."""
    rng = random.Random(seed)
    buckets: dict[str, list[int]] = {}
    for i, q in enumerate(qs):
        buckets.setdefault(q.kind, []).append(i)
    out: list[list[int]] = [[] for _ in range(n_folds)]
    for idxs in buckets.values():
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            out[j % n_folds].append(i)
    return out


# --- the tuning procedure ---------------------------------------------------


@dataclass
class Outcome:
    current: FusionParams
    best: FusionParams  # best on the full set, adopted or not
    chosen: FusionParams  # what ships: `best` if adopted, else `current`
    adopted: bool
    current_full: Score
    chosen_full: Score
    current_cv: float
    tuned_cv: float
    margin: float
    n: int
    reason: str


def tune(
    qs: list[Question],
    current: FusionParams,
    *,
    rerank: bool,
    n_folds: int = DEFAULT_FOLDS,
    conn=None,
    reranker=None,
) -> Outcome:
    kw = {"rerank": rerank, "conn": conn, "reranker": reranker}
    candidates = grid(current, rerank=rerank)
    n = len(qs)
    margin = 1.0 / n if n else 0.0

    if rerank:
        _prescore(qs, candidates, conn=conn, reranker=reranker)

    # Every candidate's score on every question, once. Cross-validation then
    # only re-averages columns — which is what makes a 10k-point grid take
    # minutes rather than an hour.
    table = [per_question(qs, p, **kw) for p in candidates]
    cur_row = per_question(qs, current, **kw)
    everything = list(range(n))

    # Honest estimate: choose on train folds, score on the held-out fold.
    tuned_cv_total = 0.0
    current_cv_total = 0.0
    for test_idx in folds(qs, n_folds):
        held = set(test_idx)
        train_idx = [i for i in everything if i not in held]
        chosen_here, _ = best_on(train_idx, table, candidates, current)
        row = table[candidates.index(chosen_here)]
        tuned_cv_total += sum(row[j] for j in test_idx)
        current_cv_total += sum(cur_row[j] for j in test_idx)
    tuned_cv = tuned_cv_total / n if n else 0.0
    current_cv = current_cv_total / n if n else 0.0

    # The parameters we would ship: best on everything.
    best, _ = best_on(everything, table, candidates, current)
    current_full = score(qs, current, **kw)
    chosen_full = score(qs, best, **kw)

    gain = tuned_cv - current_cv
    if best == current:
        adopted, reason = False, "current parameters are already the best on this corpus"
    elif gain > margin:
        adopted, reason = True, (
            f"cross-validated gain {gain:+.4f} exceeds the noise margin {margin:.4f} (1/n)"
        )
    else:
        adopted, reason = False, (
            f"cross-validated gain {gain:+.4f} is within the noise margin {margin:.4f} (1/n); "
            f"the in-sample improvement does not generalise"
        )
    return Outcome(
        current, best, best if adopted else current, adopted, current_full, chosen_full,
        current_cv, tuned_cv, margin, n, reason,
    )


# --- staleness --------------------------------------------------------------


def staleness(
    prov: Provenance | None,
    specs: list[manifest.TitleSpec],
    gold: list[dict],
    rerank_model: str | None = None,
) -> list[str]:
    """Why the recorded provenance no longer describes the live setup (empty = fresh).

    ``rerank_model`` is the cross-encoder in use; pass None when tuning hybrid
    output only, where the reranker is irrelevant.
    """
    if prov is None:
        return ["parameters have never been tuned (using code defaults)"]
    reasons = []
    if rerank_model is not None and prov.rerank_model != rerank_model:
        reasons.append(
            f"reranker changed: tuned with {prov.rerank_model or '(unrecorded)'}, "
            f"now {rerank_model}"
        )
    corpus = manifest.corpus_state(specs)
    if list(prov.corpus) != corpus:
        reasons.append(f"corpus changed: tuned on {list(prov.corpus)}, now {corpus}")
    fp = eval_run.gold_fingerprint(gold)
    if prov.gold_sha256 != fp["sha256"]:
        reasons.append(f"gold set changed: tuned on {prov.gold_n} questions, now {fp['n']}")
    if prov.analyzer_version != ANALYZER_VERSION:
        reasons.append(
            f"BM25 analyzer changed: tuned on v{prov.analyzer_version}, now v{ANALYZER_VERSION}"
        )
    return reasons


# --- report -----------------------------------------------------------------


def _fmt_params(p: FusionParams) -> str:
    return (
        f"rrf_k={p.rrf_k} bm25_weight={p.bm25_weight} "
        f"band=({p.bm25_contrast_lo}, {p.bm25_contrast_hi}) anchor={p.bm25_contrast_anchor} "
        f"rerank_candidates={p.rerank_candidates} agreement_scale={p.agreement_scale}"
    )


def print_report(o: Outcome, *, rerank: bool) -> None:
    stage = "rerank" if rerank else "hybrid"
    print(f"\ntuning {stage} output on {o.n} in-corpus questions "
          f"(objective = ½ recall@10 + ½ MRR@10)\n")
    print(f"  current: {_fmt_params(o.current)}")
    print(f"  best:    {_fmt_params(o.best)}")
    print(f"\n  {'':<22}{'current':>10}{'best':>10}")
    print(f"  {'objective (in-sample)':<22}{o.current_full.objective:>10.4f}"
          f"{o.chosen_full.objective:>10.4f}")
    print(f"  {'objective (cross-val)':<22}{o.current_cv:>10.4f}{o.tuned_cv:>10.4f}")
    print(f"  {'noise margin (1/n)':<22}{o.margin:>10.4f}")
    for kind in sorted(o.current_full.by_kind):
        c, b = o.current_full.by_kind[kind], o.chosen_full.by_kind.get(kind, {})
        print(f"\n  {kind} (n = {c['n']})")
        for m in ("recall@5", "recall@10", "mrr@10"):
            print(f"    {m:<12}{c[m]:>10.3f}{b.get(m, float('nan')):>10.3f}")
    verdict = "ADOPTED" if o.adopted else "KEPT current"
    print(f"\n  {verdict}: {o.reason}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="python -m retrieval.tune", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--manifest", type=Path, default=manifest.DEFAULT_MANIFEST)
    parser.add_argument("--gold", type=Path, default=eval_run.DEFAULT_GOLD)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--params", type=Path, default=params_mod.DEFAULT_PARAMS_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--no-rerank", action="store_true", help="Tune hybrid output only")
    parser.add_argument("--rerank-model", default=None, help="Cross-encoder to tune with")
    parser.add_argument("--dry-run", action="store_true", help="Report; do not write params")
    parser.add_argument("--force", action="store_true", help="Tune even if provenance is fresh")
    parser.add_argument("--results-dir", type=Path, default=eval_run.DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report whether the committed params match the committed corpus and gold "
        "set; exit 1 if not. Needs no index or models, so CI can run it.",
    )
    args = parser.parse_args(argv)

    from rag.reranker import DEFAULT_RERANK_MODEL

    if args.rerank_model is None:
        args.rerank_model = DEFAULT_RERANK_MODEL
    specs = manifest.load(args.manifest)
    gold = eval_run.load_gold(args.gold)
    current, prov = params_mod.load(args.params)
    reasons = staleness(prov, specs, gold, None if args.no_rerank else args.rerank_model)
    if args.check:
        # Everything staleness() reads is committed — corpus.toml, gold.jsonl,
        # params.json, the analyzer version in code — so this is decidable on a
        # bare checkout. It is the gate that makes "I forgot to re-tune"
        # impossible to merge: a new title or an edited gold set changes the
        # fingerprint, and only the tune stage can bring params.json back into
        # agreement.
        if reasons:
            print(f"STALE: {args.params} was not tuned for this corpus/gold set:")
            for r in reasons:
                print(f"  - {r}")
            print("run `uv run python -m pipeline.onboard --from tune` locally and commit "
                  "retrieval/params.json")
            return 1
        print(f"fresh: {args.params} matches corpus.toml, {args.gold}, analyzer "
              f"v{ANALYZER_VERSION} and reranker {args.rerank_model} (tuned {prov.tuned_at})")
        return 0
    if not reasons and not args.force:
        print(f"parameters are fresh for this corpus and gold set (tuned {prov.tuned_at}); "
              f"nothing to do (--force to re-tune anyway)")
        return 0
    for r in reasons:
        print(f"re-tuning because {r}")

    from ingest import fetcher

    fetcher._use_os_trust_store()
    embedder = Embedder(args.model)
    reranker = None
    if not args.no_rerank:
        from rag.reranker import Reranker

        reranker = Reranker(args.rerank_model)

    conn = connect(args.db)
    try:
        t0 = time.monotonic()
        qs = _collect(gold, conn=conn, embedder=embedder)
        print(f"fetched candidates for {len(qs)} questions in {time.monotonic() - t0:.0f}s")
        t0 = time.monotonic()
        outcome = tune(qs, current, rerank=not args.no_rerank, n_folds=args.folds,
                       conn=conn, reranker=reranker)
        print(f"searched {len(grid(current, rerank=not args.no_rerank))} parameter sets "
              f"x {args.folds} folds in {time.monotonic() - t0:.0f}s")
    finally:
        conn.close()

    print_report(outcome, rerank=not args.no_rerank)

    fp = eval_run.gold_fingerprint(gold)
    new_prov = Provenance(
        corpus=tuple(manifest.corpus_state(specs)),
        gold_sha256=fp["sha256"],
        gold_n=fp["n"],
        analyzer_version=ANALYZER_VERSION,
        tuned_at=params_mod.now(),
        objective=round(outcome.tuned_cv if outcome.adopted else outcome.current_cv, 4),
        note=("adopted: " if outcome.adopted else "kept: ") + outcome.reason,
        rerank_model=args.rerank_model if not args.no_rerank else "",
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report = args.results_dir / f"tune-{stamp}.json"
    report.write_text(json.dumps({
        "timestamp": stamp, "git_commit": eval_run.git_commit(), "rerank": not args.no_rerank,
        "reasons": reasons, "adopted": outcome.adopted, "reason": outcome.reason,
        "current": outcome.current.to_dict(), "best": outcome.best.to_dict(),
        "chosen": outcome.chosen.to_dict(),
        "current_full": outcome.current_full.__dict__, "chosen_full": outcome.chosen_full.__dict__,
        "current_cv": outcome.current_cv, "tuned_cv": outcome.tuned_cv, "margin": outcome.margin,
        "provenance": new_prov.to_dict(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {report}")

    if args.dry_run:
        print("dry run: params file not written")
        return 0
    params_mod.save(outcome.chosen, new_prov, args.params)
    params_mod.reset()
    what = "new parameters" if outcome.adopted else "provenance refreshed"
    print(f"wrote {args.params} ({what})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
