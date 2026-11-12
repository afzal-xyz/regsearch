"""Freeze everything the tuner's scoring loop needs, so it can run without models.

The problem this solves
-----------------------
Scoring one parameter set over the gold set costs two expensive things: a
query embedding plus a vector/BM25/citation lookup per question, and a
cross-encoder forward pass per candidate chunk. On CPU that is minutes. It is
also entirely independent of the fusion constants — which is why
:mod:`retrieval.tune` already fetches candidates once (``_collect``) and
cross-encoder-scores the union of every grid point's candidates once
(``_prescore``), then evaluates thousands of parameter sets as arithmetic.

The tuner throws that work away when the process exits. This module writes it
to disk instead, so a *different* process — one with no torch in it at all —
can pick it up and score parameter sets in milliseconds. That is what makes
an interactive sandbox possible without putting a 400 MB reranker inside the
web server.

What is in the file
-------------------
Per question: the two retrievers' candidate lists reduced to the four fields
:func:`retrieval.search.fuse` actually reads, the citation-lookup results, and
every cross-encoder score the grid could ask for. Chunk *text* and headings
are deliberately dropped — they exist only to be fed to the cross-encoder,
and the cross-encoder has already run. A reader of this cache can rank, and
cannot embed, retrieve or rerank. That is the point.

Also recorded: the set of fusion keys the cross-encoder scores cover. A
parameter set outside that set may surface a chunk nobody scored, so the
sandbox can refuse it instead of silently ranking against a missing score.

Provenance is a :class:`retrieval.params.Provenance`, the same record
``params.json`` carries, so :func:`retrieval.tune.staleness` can say whether
the cache still describes the live corpus, gold set, analyzer and reranker.

CLI:
    python -m retrieval.prescore            # build data/sandbox/prescore.json.gz
    python -m retrieval.prescore --no-rerank  # fusion only; no cross-encoder
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from eval import run as eval_run
from index.store import DEFAULT_DB
from index.tokenizer import ANALYZER_VERSION
from pipeline import manifest
from retrieval import params as params_mod
from retrieval import search
from retrieval import tune as tune_mod
from retrieval.params import FusionParams, Provenance

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path("data/sandbox/prescore.json.gz")

# Bump when the on-disk shape changes. An older file is then ignored rather
# than half-understood.
CACHE_VERSION = 1


# --- serialisation ----------------------------------------------------------


def _result_to_json(r: search.SearchResult) -> list:
    """Only what :func:`search.fuse` reads. Compact: this is the bulk of the file."""
    return [r.chunk_id, r.section_citation, r.score, r.retrievers]


def _result_from_json(row: list) -> search.SearchResult:
    chunk_id, section, score, retrievers = row
    return search.SearchResult(
        chunk_id=chunk_id,
        # Dropped at write time: fusion never reads them, and the only thing
        # that did — the cross-encoder — has already run. Leaving them empty
        # is honest about what this object can no longer be used for.
        citation="",
        section_citation=section,
        score=score,
        retrievers=list(retrievers),
        heading="",
        heading_path=[],
        text="",
    )


def _question_to_json(q: tune_mod.Question) -> dict:
    return {
        "text": q.text,
        "relevant": sorted(q.relevant),
        "kind": q.kind,
        "vector_hits": [_result_to_json(r) for r in q.cands.vector_hits],
        "bm25_chunks": [_result_to_json(r) for r in q.cands.bm25_chunks],
        "known": sorted(q.cands.known),
        "citing": [list(pair) for pair in q.cands.citing],
        "ce": q.ce,
    }


def _question_from_json(d: dict) -> tune_mod.Question:
    cands = search.Candidates(
        query=d["text"],
        vector_hits=[_result_from_json(r) for r in d["vector_hits"]],
        bm25_chunks=[_result_from_json(r) for r in d["bm25_chunks"]],
        known=frozenset(d["known"]),
        citing=[tuple(pair) for pair in d["citing"]],
    )
    q = tune_mod.Question(d["text"], set(d["relevant"]), d["kind"], cands)
    for r in cands.vector_hits + cands.bm25_chunks:
        q.chunks.setdefault(r.chunk_id, r)
    q.ce = dict(d["ce"])
    return q


@dataclass(frozen=True)
class PrescoreCache:
    """Frozen retrieval state: enough to rank, not enough to retrieve."""

    questions: list[tune_mod.Question]
    provenance: Provenance
    base_params: FusionParams
    fusion_keys: frozenset[tuple]  # fusion keys whose rerank candidates are scored
    reranked: bool  # False when built with --no-rerank: no cross-encoder scores
    warmed_at: str
    n_pairs: int

    @property
    def n_questions(self) -> int:
        return len(self.questions)

    def covers(self, p: FusionParams) -> bool:
        """Is every chunk this parameter set could rerank already scored?"""
        return tune_mod._fusion_key(p) in self.fusion_keys


def dump(cache: PrescoreCache, path: Path | str = DEFAULT_CACHE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "warmed_at": cache.warmed_at,
        "n_pairs": cache.n_pairs,
        "reranked": cache.reranked,
        "pool": search.POOL,
        "max_rerank": tune_mod.MAX_RERANK,
        "base_params": cache.base_params.to_dict(),
        "provenance": cache.provenance.to_dict(),
        "fusion_keys": [list(k) for k in sorted(cache.fusion_keys)],
        "questions": [_question_to_json(q) for q in cache.questions],
    }
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)
    return path


class CacheError(RuntimeError):
    """The cache is absent, unreadable, or written by another version."""


def load(path: Path | str = DEFAULT_CACHE) -> PrescoreCache:
    path = Path(path)
    if not path.exists():
        raise CacheError(f"no prescore cache at {path} — warm it first")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        raise CacheError(f"could not read {path}: {exc}") from exc
    if payload.get("version") != CACHE_VERSION:
        raise CacheError(
            f"{path} was written by cache version {payload.get('version')}, "
            f"this is version {CACHE_VERSION} — re-warm it"
        )
    return PrescoreCache(
        questions=[_question_from_json(d) for d in payload["questions"]],
        provenance=Provenance.from_dict(payload["provenance"]),
        base_params=FusionParams.from_dict(payload["base_params"]),
        fusion_keys=frozenset(tuple(k) for k in payload["fusion_keys"]),
        reranked=bool(payload.get("reranked", True)),
        warmed_at=str(payload.get("warmed_at", "")),
        n_pairs=int(payload.get("n_pairs", 0)),
    )


# --- building ---------------------------------------------------------------


def build(
    gold: list[dict],
    specs: list[manifest.TitleSpec],
    *,
    conn,
    embedder,
    reranker=None,
    base_params: FusionParams | None = None,
    rerank: bool = True,
) -> PrescoreCache:
    """Run the tuner's own collect/prescore and capture the result.

    Deliberately calls :func:`retrieval.tune._collect` and
    :func:`retrieval.tune._prescore` rather than reimplementing them: the
    sandbox's numbers have to be the tuner's numbers, and the surest way to
    guarantee that is for the same code to produce the candidates.
    """
    base = base_params or params_mod.current()
    questions = tune_mod._collect(gold, conn=conn, embedder=embedder)
    candidates = tune_mod.grid(base, rerank=rerank)
    keys = {tune_mod._fusion_key(p) for p in candidates}
    if rerank:
        if reranker is None:
            raise ValueError("rerank=True needs a reranker")
        tune_mod._prescore(questions, candidates, conn=conn, reranker=reranker)
    return PrescoreCache(
        questions=questions,
        provenance=Provenance(
            corpus=tuple(manifest.corpus_state(specs)),
            gold_sha256=eval_run.gold_fingerprint(gold)["sha256"],
            gold_n=len(gold),
            analyzer_version=ANALYZER_VERSION,
            tuned_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            note="prescore cache for the parameter sandbox; not a tuning result",
            rerank_model=getattr(reranker, "model_name", "") if rerank else "",
        ),
        base_params=base,
        fusion_keys=frozenset(keys),
        reranked=rerank,
        warmed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_pairs=sum(len(q.ce) for q in questions),
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="python -m retrieval.prescore", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--manifest", type=Path, default=manifest.DEFAULT_MANIFEST)
    parser.add_argument("--gold", type=Path, default=eval_run.DEFAULT_GOLD)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", default=None, help="Embedding model")
    parser.add_argument("--rerank-model", default=None, help="Cross-encoder")
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Skip the cross-encoder: fusion-only sandbox, warms in seconds",
    )
    args = parser.parse_args(argv)

    from index.embedder import DEFAULT_MODEL, Embedder
    from index.store import connect
    from ingest import fetcher
    from rag import reranker as reranker_mod

    fetcher._use_os_trust_store()
    specs = manifest.load(args.manifest)
    gold = eval_run.load_gold(args.gold)
    embedder = Embedder(args.model or DEFAULT_MODEL)
    reranker = (
        None if args.no_rerank
        else reranker_mod.Reranker(args.rerank_model or reranker_mod.DEFAULT_RERANK_MODEL)
    )

    # Preflight: load both models now, so a broken ML stack (e.g. a DLL that
    # Windows Smart App Control refuses) fails here with one clear message
    # instead of mid-run inside retrieval.
    try:
        embedder.model
        if reranker is not None:
            reranker.model
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(
        f"warming the sandbox cache: {len(gold)} gold questions, "
        f"{len(specs)} titles, rerank={not args.no_rerank}"
    )
    conn = connect(args.db)
    try:
        cache = build(
            gold, specs, conn=conn, embedder=embedder, reranker=reranker,
            rerank=not args.no_rerank,
        )
    finally:
        conn.close()

    out = dump(cache, args.out)
    size = out.stat().st_size / 1e6
    print(
        f"\nwrote {out} ({size:.1f} MB): {cache.n_questions} questions, "
        f"{cache.n_pairs} cross-encoder scores, {len(cache.fusion_keys)} fusion points covered"
    )
    print("the sandbox can now score any grid point in milliseconds, with no model loaded")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
