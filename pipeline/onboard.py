"""End-to-end corpus onboarding: fetch → chunk → index → gate → gold → eval.

One command takes a CFR title from "not in the repo" to "indexed, verified,
measured, and guarded against regression":

    python -m pipeline.onboard --title 1 --date 2024-07-18 --name "General Provisions"

**Every stage runs over the whole declared corpus, not just the new title.**
That is the central design choice and it is deliberate. A per-title pipeline
produces a corpus that is correct title-by-title and broken as a whole: Title
1 chunked by a newer parser than Title 4 was, an index holding a title the
manifest forgot, a gold set that covers two of three titles. Those are exactly
the failures nobody notices, because every individual command succeeded. So
the unit of work is the corpus, the manifest is its definition, and a stage
either brings all of it up to date or fails.

The cost is that onboarding re-walks titles that are already done. Every stage
is idempotent and cheap when there is nothing to do — fetch is a cache hit,
chunk is skipped unless the raw XML is newer, index is skipped unless the
chunk file is newer — so in practice re-walking two finished titles costs
about a second, and the guarantee is worth far more than that.

Stages, in order:

0. ``preflight`` — fail fast on what can be checked before any download: the
   manifest parses, the gold file parses, both models load, the params file
   is readable. Finding these out after a ten-minute index build is the
   difference between a pipeline people run and one they babysit.
1. ``fetch``     — download each title's XML, retrying transient eCFR failures.
2. ``chunk``     — parse and chunk each title to JSONL.
3. ``index``     — embed and index each title (idempotent per title+date).
4. ``integrity`` — gate the corpus: chunks, index, retrieval probe, and
   whether the retrieval parameters were tuned for *this* corpus.
5. ``gold``      — gate gold-set schema and per-title coverage.
6. ``tune``      — re-tune fusion/rerank parameters against the current
   corpus and gold set, cross-validated, adopting a change only when it
   beats the current parameters by more than noise. Runs before ``eval`` so
   the recorded numbers are for the parameters that will actually be used.
7. ``eval``      — score every retrieval mode against the gold set.
8. ``regression``— diff against the previous run and fail on a real drop.

CLI:
    python -m pipeline.onboard                      # bring the declared corpus up to date
    python -m pipeline.onboard --title 1 --date 2024-07-18
    python -m pipeline.onboard --only integrity     # one stage
    python -m pipeline.onboard --from index         # that stage onward
    python -m pipeline.onboard --allow-gold-gaps    # during gold authoring
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from index.store import DEFAULT_DB
from pipeline import manifest
from pipeline.manifest import ManifestError, TitleSpec

logger = logging.getLogger(__name__)

STAGES = ("preflight", "fetch", "chunk", "index", "integrity", "gold", "tune", "eval", "regression")


class StageFailed(RuntimeError):
    """A stage failed; later stages must not run on a corpus in this state."""


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def _step(text: str) -> None:
    print(f"  - {text}", flush=True)


# --- stages -----------------------------------------------------------------


def stage_preflight(
    specs: list[TitleSpec], *, gold_path: Path | None = None, skip_models: bool = False, **kw
) -> None:
    """Everything cheap that would otherwise fail a long way into the run.

    Each check here was a real late failure or an obvious one waiting to
    happen: a gold file with a JSON typo that `eval` finds after indexing; a
    model download that times out after `fetch` spent five minutes; a
    corrupt params file that silently reverts search to defaults.
    """
    from eval import goldset
    from retrieval import params as fusion_params

    dates = {s.date for s in specs}
    _step(f"manifest: {len(specs)} title(s), date(s) {sorted(dates)}")
    if len(dates) > 1:
        # Legal, but the corpus is then a mix of vintages: two titles that
        # cross-reference each other may be quoting different text.
        _step("WARNING: titles are declared at different issue dates; the corpus is not a "
              "single snapshot")

    gold_path = gold_path or goldset.DEFAULT_GOLD
    try:
        entries = goldset.load(gold_path)
    except ValueError as exc:
        raise StageFailed(f"gold file: {exc}") from exc
    schema = [p for p in goldset.validate(entries) if p.severity == "error"]
    if schema:
        for p in schema[:5]:
            _step(str(p))
        raise StageFailed(f"gold file has {len(schema)} schema error(s)")
    _step(f"gold file: {len(entries)} entries parse and validate")

    params_path = fusion_params.DEFAULT_PARAMS_PATH
    if params_path.exists():
        _, prov = fusion_params.load(params_path)
        if prov is None:
            raise StageFailed(
                f"{params_path} exists but is unreadable or has no provenance; search would "
                f"silently fall back to defaults. Fix or delete it."
            )
        _step(f"params: {params_path} (tuned {prov.tuned_at})")
    else:
        _step("params: none yet; the tune stage will create retrieval/params.json")

    if skip_models:
        _step("models: skipped (--skip-models)")
        return
    from index.embedder import Embedder
    from ingest import fetcher
    from rag.reranker import Reranker

    fetcher._use_os_trust_store()
    start = time.monotonic()
    try:
        emb = Embedder()
        emb.encode_query("preflight")
        Reranker().score("preflight", ["preflight"])
    except Exception as exc:  # noqa: BLE001 - any model failure must stop the run
        raise StageFailed(f"models: {exc}") from exc
    _step(f"models: embedder ({emb.model_name}, dim {emb.dim}) and cross-encoder load "
          f"({time.monotonic() - start:.0f}s)")


def stage_fetch(specs: list[TitleSpec], *, force: bool = False, **kw) -> None:
    """Download every declared title's XML. Cached titles cost nothing."""
    from ingest import fetcher

    fetcher._use_os_trust_store()
    for spec in specs:
        path = spec.raw_path()
        if path.exists() and not force:
            _step(f"{spec.label}: cached ({path.stat().st_size / 1e6:.1f} MB)")
            continue
        _step(f"{spec.label}: downloading (retries on transient eCFR failures)…")
        start = time.monotonic()
        try:
            path = fetcher.fetch_title_xml(spec.number, spec.date, force=force)
        except Exception as exc:
            raise StageFailed(f"fetch {spec.label}: {exc}") from exc
        _step(
            f"{spec.label}: {path.stat().st_size / 1e6:.1f} MB "
            f"in {time.monotonic() - start:.0f}s -> {path}"
        )


def _newer(a: Path, b: Path) -> bool:
    """True when ``a`` exists and is strictly newer than ``b`` (or ``b`` is absent)."""
    if not a.exists():
        return False
    return not b.exists() or a.stat().st_mtime > b.stat().st_mtime


def stage_chunk(specs: list[TitleSpec], *, force: bool = False, **kw) -> None:
    """Parse and chunk every declared title whose chunk file is missing or stale."""
    from ingest.chunker import chunk_title

    for spec in specs:
        raw, chunks = spec.raw_path(), spec.chunks_path()
        if chunks.exists() and not force and not _newer(raw, chunks):
            n = sum(1 for line in chunks.open(encoding="utf-8") if line.strip())
            _step(f"{spec.label}: up to date ({n} chunks)")
            continue
        _step(f"{spec.label}: chunking…")
        try:
            out, n = chunk_title(spec.number, spec.date, force=force)
        except Exception as exc:
            raise StageFailed(f"chunk {spec.label}: {exc}") from exc
        _step(f"{spec.label}: {n} chunks -> {out}")


def stage_index(
    specs: list[TitleSpec], *, db_path: Path = DEFAULT_DB, force: bool = False, **kw
) -> None:
    """Embed and index every declared title whose chunks are newer than the index.

    ``build_index`` is idempotent per (title, date) — it deletes that title's
    rows before inserting — so re-indexing is safe but costs an embedding pass
    over the whole title. The mtime check is what keeps a routine run cheap.
    """
    import sqlite3

    from index.store import build_index, connect
    from ingest import fetcher

    fetcher._use_os_trust_store()

    indexed: dict[int, tuple[str, int]] = {}
    if Path(db_path).exists():
        try:
            conn = connect(db_path)
            try:
                indexed = {
                    int(t): (d, int(n))
                    for t, d, n in conn.execute(
                        "select title, date, count(*) from chunks group by title, date"
                    )
                }
            finally:
                conn.close()
        except sqlite3.Error:
            indexed = {}

    db = Path(db_path)
    for spec in specs:
        chunks = spec.chunks_path()
        have = indexed.get(spec.number)
        on_disk = sum(1 for line in chunks.open(encoding="utf-8") if line.strip())
        fresh = have is not None and have[0] == spec.date and have[1] == on_disk
        if fresh and not force and not _newer(chunks, db):
            _step(f"{spec.label}: already indexed ({have[1]} chunks)")
            continue
        _step(f"{spec.label}: embedding {on_disk} chunks (first run downloads the model)…")
        start = time.monotonic()
        try:
            summary = build_index(chunks, db_path=db_path, show_progress=False)
        except Exception as exc:
            raise StageFailed(f"index {spec.label}: {exc}") from exc
        _step(
            f"{spec.label}: +{summary['added']} chunks, {summary['total']} total "
            f"in {time.monotonic() - start:.0f}s"
        )


def stage_integrity(
    specs: list[TitleSpec], *, db_path: Path = DEFAULT_DB, strict: bool = False, **kw
) -> None:
    from pipeline import integrity

    problems = integrity.run_checks(specs, db_path=db_path)
    ok, text = integrity.summarize(problems, strict=strict)
    print(text)
    if not ok:
        raise StageFailed("integrity gate failed")


def stage_gold(
    specs: list[TitleSpec],
    *,
    db_path: Path = DEFAULT_DB,
    gold_path: Path | None = None,
    allow_gold_gaps: bool = False,
    strict: bool = False,
    **kw,
) -> None:
    """Gate the gold set. ``allow_gold_gaps`` downgrades coverage errors to warnings.

    The escape hatch exists for one specific, legitimate window: a title is
    indexed and its draft questions are being written. Without it the pipeline
    would be unrunnable precisely when you need it most — you cannot write
    good prose questions about a title you cannot search. It is *not* a way to
    ship an unmeasured title, and the stage says so every time it is used.
    """
    from eval import goldset

    gold_path = gold_path or goldset.DEFAULT_GOLD
    problems = goldset.run_checks(specs, gold_path=gold_path, db_path=db_path)
    coverage_errors = [p for p in problems if p.check.startswith("gold:coverage")]
    other_errors = [
        p for p in problems if p.severity == "error" and not p.check.startswith("gold:coverage")
    ]
    warnings = [p for p in problems if p.severity == "warning"]

    for p in problems:
        print(f"  {p}")
    goldset._print_coverage(goldset.coverage(goldset.load(gold_path), specs), specs)

    if coverage_errors and allow_gold_gaps:
        print(
            "\n  NOTE: --allow-gold-gaps is set, so the coverage shortfalls above are not "
            "failing the run.\n"
            "  Those titles are indexed but UNMEASURED: their retrieval quality is not in "
            "any metric,\n"
            "  and a regression in them cannot be detected. Finish the gold set before "
            "treating this corpus as done.",
        )
        coverage_errors = []

    if other_errors or coverage_errors or (strict and warnings):
        raise StageFailed("gold-set gate failed")
    if not problems:
        print("\n  gold set: all checks passed")


def stage_tune(
    specs: list[TitleSpec],
    *,
    db_path: Path = DEFAULT_DB,
    gold_path: Path | None = None,
    skip_tune: bool = False,
    **kw,
) -> None:
    """Bring the retrieval parameters into agreement with the current corpus.

    Cheap when nothing changed: the params file records what it was tuned on,
    and if that matches the live corpus, gold set and analyzer the stage is a
    no-op. Otherwise it cross-validates a grid and adopts a change only when
    the held-out gain exceeds one question's worth (see :mod:`retrieval.tune`).
    """
    from retrieval import tune

    if skip_tune:
        _step("skipped (--skip-tune); the integrity gate will keep warning if params are stale")
        return
    argv = ["--db", str(db_path)]
    if gold_path:
        argv += ["--gold", str(gold_path)]
    if tune.main(argv) != 0:
        raise StageFailed("tuning failed")


def stage_eval(specs: list[TitleSpec], *, db_path: Path = DEFAULT_DB, gold_path=None, **kw) -> None:
    from eval import run as eval_run

    argv = ["--db", str(db_path)]
    if gold_path:
        argv += ["--gold", str(gold_path)]
    if eval_run.main(argv) != 0:
        raise StageFailed("eval run failed")


def stage_regression(specs: list[TitleSpec], *, no_fail: bool = False, **kw) -> None:
    """Diff the run just written against the previous one and fail on a drop.

    Onboarding a title adds its questions to the gold set, so the first run
    afterwards averages over a different population than the one before it
    and the deltas are not evidence of anything. Each run records a
    fingerprint of its question set and ``eval/compare.py`` detects the
    change, prints the deltas for information, and declines to fail on them
    (see :func:`eval.compare.comparable`). The gate bites from the second run
    onward, which is when it should.
    """
    from eval import compare

    argv = ["--no-fail"] if no_fail else []
    if compare.main(argv) != 0:
        raise StageFailed("regression gate failed")


STAGE_FUNCS = {
    "preflight": stage_preflight,
    "fetch": stage_fetch,
    "chunk": stage_chunk,
    "index": stage_index,
    "integrity": stage_integrity,
    "gold": stage_gold,
    "tune": stage_tune,
    "eval": stage_eval,
    "regression": stage_regression,
}


# --- driver -----------------------------------------------------------------


def select_stages(only: str | None, start: str | None) -> list[str]:
    if only:
        return [only]
    if start:
        return list(STAGES[STAGES.index(start) :])
    return list(STAGES)


def run(specs: list[TitleSpec], stages: list[str], **opts) -> int:
    print(
        f"corpus: {len(specs)} title(s) — "
        + ", ".join(f"{s.number}@{s.date}" for s in specs)
    )
    print(f"stages: {' -> '.join(stages)}")
    for name in stages:
        _banner(f"stage: {name}   (over all {len(specs)} declared title(s))")
        try:
            STAGE_FUNCS[name](specs, **opts)
        except StageFailed as exc:
            print(f"\nFAILED at stage '{name}': {exc}", file=sys.stderr)
            remaining = stages[stages.index(name) + 1 :]
            if remaining:
                print(f"not running: {', '.join(remaining)}", file=sys.stderr)
            return 1
    _banner("onboarding complete")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="python -m pipeline.onboard",
        description="Fetch, chunk, index, gate and evaluate the whole declared corpus.",
    )
    parser.add_argument("--title", type=int, help="CFR title to add to the manifest first")
    parser.add_argument("--date", help="Issue date for --title (YYYY-MM-DD)")
    parser.add_argument("--name", default="", help="Human-readable title name")
    parser.add_argument("--manifest", type=Path, default=manifest.DEFAULT_MANIFEST)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--gold", type=Path, help="Gold set path (default eval/gold.jsonl)")
    parser.add_argument("--only", choices=STAGES, help="Run just this stage")
    parser.add_argument("--from", dest="start", choices=STAGES, help="Run from this stage on")
    parser.add_argument("--force", action="store_true", help="Redo fetch/chunk/index even if fresh")
    parser.add_argument("--strict", action="store_true", help="Treat gate warnings as failures")
    parser.add_argument(
        "--allow-gold-gaps",
        action="store_true",
        help="Report gold coverage shortfalls without failing (while authoring questions)",
    )
    parser.add_argument(
        "--no-fail", action="store_true", help="Report regressions without a non-zero exit"
    )
    parser.add_argument("--skip-tune", action="store_true", help="Do not re-tune parameters")
    parser.add_argument(
        "--skip-models", action="store_true", help="Preflight: do not load the models"
    )
    args = parser.parse_args(argv)

    if args.title is not None:
        if not args.date:
            print("error: --title requires --date", file=sys.stderr)
            return 1
        try:
            specs, added = manifest.add(
                args.title, args.date, args.name, path=args.manifest
            )
        except (ManifestError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"manifest: {'added' if added else 'already declared'} "
            f"Title {args.title} @ {args.date}"
        )
    else:
        try:
            specs = manifest.load(args.manifest)
        except ManifestError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    return run(
        specs,
        select_stages(args.only, args.start),
        db_path=args.db,
        gold_path=args.gold,
        force=args.force,
        strict=args.strict,
        allow_gold_gaps=args.allow_gold_gaps,
        no_fail=args.no_fail,
        skip_tune=args.skip_tune,
        skip_models=args.skip_models,
    )


if __name__ == "__main__":
    raise SystemExit(main())
