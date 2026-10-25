"""Evaluation harness: score vector-only, bm25-only, and hybrid retrieval against
``eval/gold.jsonl``.

Metrics (per mode, over the in-corpus questions only):
- recall@5, recall@10 — did any retrieved chunk's section land in the top k?
- MRR@10          — mean reciprocal rank of the first section-level hit.

A "hit" = a retrieved chunk whose ``section_citation`` is in the question's
``relevant_citations``. Negatives (empty ``relevant_citations``) are excluded
from recall/MRR and reported separately; each question's per-mode first-hit rank
(or null) is recorded in the results JSON for later analysis.

CLI:
    python -m eval.run [--mode all|vector|bm25|hybrid] [--k 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from index.embedder import DEFAULT_MODEL, Embedder
from index.store import DEFAULT_DB, connect
from retrieval import search

logger = logging.getLogger(__name__)

DEFAULT_GOLD = Path("eval/gold.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results")
MODES = ("vector", "bm25", "hybrid")


# --- pure metric functions (unit-tested in isolation) ----------------------


def first_hit_rank(ranked: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first section in ``ranked`` that is relevant, else None."""
    for i, section in enumerate(ranked, start=1):
        if section in relevant:
            return i
    return None


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    """1.0 if any of the top-k ranked sections is relevant, else 0.0.

    Returns None when ``relevant`` is empty (recall is undefined — a negative)."""
    if not relevant:
        return None
    return 1.0 if any(section in relevant for section in ranked[:k]) else 0.0


def reciprocal_rank_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    """1 / rank of the first relevant section within the top k, else 0.0 (None if negative)."""
    if not relevant:
        return None
    rank = first_hit_rank(ranked[:k], relevant)
    return 1.0 / rank if rank else 0.0


# --- retrieval per mode -----------------------------------------------------


def _dedupe_sections(results: list[search.SearchResult], k: int) -> list[str]:
    """Collapse chunk-level results to an ordered list of distinct section citations."""
    ordered: list[str] = []
    for r in results:
        if r.section_citation not in ordered:
            ordered.append(r.section_citation)
        if len(ordered) >= k:
            break
    return ordered


def retrieve(mode: str, question: str, k: int, *, conn, embedder) -> list[str]:
    """Return up to ``k`` distinct section citations for ``mode``, best first."""
    pool = k * 4  # over-fetch chunks so k distinct sections survive de-duping
    if mode == "vector":
        results = search.vector_search(question, pool, conn=conn, embedder=embedder)
    elif mode == "bm25":
        results = search.bm25_search(question, pool, conn=conn)
    elif mode == "hybrid":
        results = search.hybrid_search(question, pool, conn=conn, embedder=embedder)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return _dedupe_sections(results, k)


# --- evaluation loop --------------------------------------------------------


def load_gold(path: Path | str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    gold = [json.loads(line) for line in lines if line.strip()]
    approved = [g for g in gold if g.get("status", "approved") == "approved"]
    skipped = len(gold) - len(approved)
    if skipped:
        logger.warning("skipping %d gold entries not marked approved", skipped)
    return approved


METRICS = ("recall@5", "recall@10", "mrr@10")


def _aggregate(scored: list[dict], modes) -> dict:
    """Mean of each metric per mode over ``scored`` (one entry per question)."""
    summary = {}
    for m in modes:
        summary[m] = {
            metric: (
                sum(s["scores"][m][metric] for s in scored) / len(scored) if scored else None
            )
            for metric in METRICS
        }
        summary[m]["n"] = len(scored)
    return summary


def evaluate(gold: list[dict], modes, *, conn, embedder, k: int = 10) -> dict:
    in_corpus = [g for g in gold if g["relevant_citations"]]
    negatives = [g for g in gold if not g["relevant_citations"]]

    scored: list[dict] = []
    details = []

    for g in gold:
        relevant = set(g["relevant_citations"])
        # Questions phrased in everyday language and known-item citation
        # lookups are different tasks with very different difficulty; mixing
        # them into one number hides movement in both.
        kind = g.get("kind", "prose")
        entry = {
            "question": g["question"],
            "relevant_citations": g["relevant_citations"],
            "kind": kind,
            "modes": {},
        }
        per_mode = {}
        for m in modes:
            ranked = retrieve(m, g["question"], k, conn=conn, embedder=embedder)
            entry["modes"][m] = {
                "first_hit_rank": first_hit_rank(ranked, relevant),  # null for negatives/misses
                "top_sections": ranked,
            }
            if relevant:  # score in-corpus questions only
                per_mode[m] = {
                    "recall@5": recall_at_k(ranked, relevant, 5),
                    "recall@10": recall_at_k(ranked, relevant, 10),
                    "mrr@10": reciprocal_rank_at_k(ranked, relevant, 10),
                }
        if relevant:
            scored.append({"kind": kind, "scores": per_mode})
        details.append(entry)

    kinds = sorted({s["kind"] for s in scored})
    return {
        "k": k,
        "n_in_corpus": len(in_corpus),
        "n_negatives": len(negatives),
        "summary": _aggregate(scored, modes),
        "summary_by_kind": {
            kind: _aggregate([s for s in scored if s["kind"] == kind], modes) for kind in kinds
        },
        "details": details,
    }


def _print_table(summary: dict, title: str = "") -> None:
    header = f"{'mode':<8} {'recall@5':>9} {'recall@10':>10} {'MRR@10':>8} {'n':>4}"
    if title:
        print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for mode, m in summary.items():
        print(
            f"{mode:<8} {m['recall@5']:>9.3f} {m['recall@10']:>10.3f} "
            f"{m['mrr@10']:>8.3f} {m['n']:>4}"
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="python -m eval.run")
    parser.add_argument("--mode", choices=["all", *MODES], default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args(argv)

    modes = list(MODES) if args.mode == "all" else [args.mode]
    gold = load_gold(args.gold)

    # Only load the embedding model if a semantic mode is in play.
    embedder = None
    if any(m in ("vector", "hybrid") for m in modes):
        from ingest import fetcher

        fetcher._use_os_trust_store()
        embedder = Embedder(args.model)

    conn = connect(args.db)
    try:
        result = evaluate(gold, modes, conn=conn, embedder=embedder, k=args.k)
    finally:
        conn.close()

    _print_table(result["summary"], "ALL in-corpus questions")
    by_kind = result["summary_by_kind"]
    if len(by_kind) > 1:
        for kind, summary in by_kind.items():
            _print_table(summary, f"kind = {kind}")
    print(
        f"\nin-corpus questions: {result['n_in_corpus']}   "
        f"negatives (excluded from metrics): {result['n_negatives']}"
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.results_dir / f"{stamp}.json"
    payload = {
        "timestamp": stamp,
        "gold": str(args.gold),
        "db": args.db,
        "model": args.model,
        "modes": modes,
        **result,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
