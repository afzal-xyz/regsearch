"""Integrity gate: refuse to ship a corpus that is subtly broken.

Every check here exists because the failure it catches is silent — the
pipeline completes, the index builds, search returns results, and the results
are wrong or incomplete in a way no exception reports. The gate is run after
chunking and indexing and exits non-zero on any error, so CI and
:mod:`pipeline.onboard` both block on it.

The checks fall in three layers:

- **chunk** — the JSONL a title chunks to: schema, non-empty text, and the
  citation metadata CLAUDE.md requires every chunk to carry.
- **index** — the SQLite database against the manifest: every declared title
  present at the declared date, nothing present that is *not* declared,
  counts matching the chunk files, one vector per chunk, and a BM25 index
  built by the current analyzer.
- **retrieval** — a functional probe. The layers above can all pass on an
  index that cannot actually find anything, so a sample of sections is
  looked up by their own citation and must come back first.

CLI:
    python -m pipeline.integrity            # gate the whole declared corpus
    python -m pipeline.integrity --strict   # treat warnings as errors too
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from index.store import DEFAULT_DB, connect, get_meta, load_bm25
from index.tokenizer import ANALYZER_VERSION
from pipeline import manifest
from pipeline.manifest import TitleSpec

logger = logging.getLogger(__name__)

ERROR = "error"
WARNING = "warning"

# Fields every chunk must carry. `text` and the citation pair are the ones
# that matter for correctness; the rest make a chunk traceable to its source.
REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "citation",
    "section_citation",
    "heading_path",
    "title",
    "date",
    "text",
)

# The two shapes a full CFR citation path takes — CLAUDE.md requires every
# chunk to carry one. Anything else means the parser mis-read the hierarchy.
#
#   section:   "4 CFR § 7.3", "31 CFR § 1010.311(a)(2)"
#   appendix:  "4 CFR Appendix I to Part 83"
#              "31 CFR Appendix A to Subpart C of Part 1010"
#              "31 CFR Appendix to Part 50"        (unlettered)
#              "31 CFR Appendix A to Chapter V"    (attached to a chapter)
#
# Appendices are not an edge case to tolerate: 358 of this corpus's chunks are
# appendix material, and an appendix is often where a rule's actual tables and
# dollar thresholds live. A gate that rejected them would be rejecting real
# regulation text.
#
# The container after "to" is deliberately *not* enumerated. eCFR supplies that
# string verbatim in the appendix's heading, so every list of containers is a
# list of the ones seen so far — a first draft of this check allowed only
# "Part N" and failed on Title 4's appendices, and the next draft allowed
# "Subpart X of Part N" too and failed on "Appendix A to Chapter V". What the
# check can honestly assert is the *shape*: a title number, "CFR", and either a
# § citation or an appendix naming some container.
_CITATION_RE = re.compile(r"^(\d+) CFR (?:§ \S+|Appendix(?: \S+)? to \S+)")

# Sections probed by the retrieval check. Enough to catch a dead BM25 index or
# a tokenizer regression without turning the gate into an eval run.
DEFAULT_PROBE_SECTIONS = 25
PROBE_SEED = 20260730


@dataclass(frozen=True)
class Problem:
    check: str
    severity: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity:<7}] {self.check}: {self.message}"


def _err(check: str, message: str) -> Problem:
    return Problem(check, ERROR, message)


def _warn(check: str, message: str) -> Problem:
    return Problem(check, WARNING, message)


# --- chunk-level ------------------------------------------------------------


def _read_chunks(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def check_chunk_file(spec: TitleSpec, chunks: list[dict]) -> list[Problem]:
    """Schema, text and citation checks for one title's chunk JSONL.

    Problems are reported once per kind with a count and one example, not once
    per chunk: a parser bug produces thousands of identical failures and a
    screen of them hides every other check.
    """
    problems: list[Problem] = []
    if not chunks:
        return [_err("chunk:empty", f"{spec.label}: chunk file has no chunks")]

    missing: dict[str, list[str]] = {}
    empty_text: list[str] = []
    bad_citation: list[str] = []
    wrong_title: list[str] = []
    wrong_date: list[str] = []
    empty_path: list[str] = []

    for c in chunks:
        cid = str(c.get("chunk_id", "<no chunk_id>"))
        for field in REQUIRED_CHUNK_FIELDS:
            if field not in c or c[field] in (None, ""):
                missing.setdefault(field, []).append(cid)
        if not str(c.get("text", "")).strip():
            empty_text.append(cid)
        if not c.get("heading_path"):
            empty_path.append(cid)
        for field in ("citation", "section_citation"):
            m = _CITATION_RE.match(str(c.get(field, "")))
            if not m:
                bad_citation.append(f"{cid} {field}={c.get(field)!r}")
            elif int(m.group(1)) != spec.number:
                wrong_title.append(f"{cid} {field}={c.get(field)!r}")
        if c.get("title") != spec.number:
            wrong_title.append(f"{cid} title={c.get('title')!r}")
        if c.get("date") != spec.date:
            wrong_date.append(f"{cid} date={c.get('date')!r}")

    def report(check: str, bad: list[str], what: str) -> None:
        if bad:
            problems.append(
                _err(check, f"{spec.label}: {len(bad)} chunk(s) {what} (e.g. {bad[0]})")
            )

    for field, bad in sorted(missing.items()):
        report(f"chunk:missing-{field}", bad, f"missing `{field}`")
    report("chunk:empty-text", empty_text, "have empty text")
    report("chunk:heading-path", empty_path, "have an empty heading_path")
    report("chunk:citation-format", bad_citation, "have a malformed citation")
    report("chunk:title-mismatch", wrong_title, f"do not belong to title {spec.number}")
    report("chunk:date-mismatch", wrong_date, f"are not dated {spec.date}")

    ids = [c.get("chunk_id") for c in chunks]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        problems.append(
            _err(
                "chunk:duplicate-id",
                f"{spec.label}: {len(dupes)} duplicate chunk_id(s) "
                f"(e.g. {sorted(dupes)[0]}) — indexing would silently drop chunks",
            )
        )
    return problems


# --- index-level ------------------------------------------------------------


def check_index(
    conn: sqlite3.Connection,
    specs: list[TitleSpec],
    counts: dict[int, int],
) -> list[Problem]:
    """The database against the manifest. ``counts`` is chunks-per-title on disk."""
    problems: list[Problem] = []

    rows = conn.execute("select title, date, count(*) from chunks group by title, date").fetchall()
    in_db: dict[int, list[tuple[str, int]]] = {}
    for title, date, n in rows:
        in_db.setdefault(int(title), []).append((date, int(n)))

    declared = {s.number: s for s in specs}

    for spec in specs:
        entries = in_db.get(spec.number)
        if not entries:
            problems.append(
                _err("index:missing-title", f"{spec.label} is declared but not in the index")
            )
            continue
        dates = {d for d, _ in entries}
        if dates != {spec.date}:
            # Two vintages of one title double every section under slightly
            # different text; retrieval returns both and the eval set can hit
            # either, which looks like flaky ranking rather than a bad index.
            problems.append(
                _err(
                    "index:date-mismatch",
                    f"{spec.label}: manifest says {spec.date}, index holds {sorted(dates)}",
                )
            )
            continue
        n_db = sum(n for _, n in entries)
        n_disk = counts.get(spec.number)
        if n_disk is not None and n_db != n_disk:
            problems.append(
                _err(
                    "index:count-mismatch",
                    f"{spec.label}: {n_db} chunks indexed but {n_disk} in the chunk file "
                    f"— the index is stale, re-run the index stage",
                )
            )

    for title in sorted(set(in_db) - set(declared)):
        n = sum(n for _, n in in_db[title])
        problems.append(
            _err(
                "index:undeclared-title",
                f"Title {title} ({n} chunks) is in the index but not in the manifest "
                f"— it will be searched but never evaluated",
            )
        )

    n_chunks = conn.execute("select count(*) from chunks").fetchone()[0]
    n_vec = conn.execute("select count(*) from vec_chunks").fetchone()[0]
    if n_chunks != n_vec:
        problems.append(
            _err(
                "index:vector-count",
                f"{n_chunks} chunks but {n_vec} vectors — vector search would miss chunks",
            )
        )

    meta = get_meta(conn)
    if not meta.get("model_name"):
        problems.append(_err("index:meta", "index has no model_name; it was never initialized"))

    # Read the persisted analyzer version *before* load_bm25, which silently
    # self-heals a stale index by rebuilding it. Self-healing is right at query
    # time but wrong here: a stale index means the BM25 confidence thresholds in
    # retrieval/search.py were calibrated against a different tokenizer, and the
    # fix is to rebuild deliberately and re-tune, not to let the first query
    # absorb it.
    stored = stored_analyzer_version(conn)
    if stored is not None and stored != ANALYZER_VERSION:
        problems.append(
            _warn(
                "index:analyzer",
                f"BM25 was built by analyzer v{stored}, current is v{ANALYZER_VERSION}; "
                f"re-run the index stage and re-check the confidence thresholds "
                f"in retrieval/search.py (they are calibrated per analyzer)",
            )
        )

    try:
        ids, model = load_bm25(conn)
    except sqlite3.Error as exc:
        problems.append(_err("index:bm25", f"BM25 index is unreadable: {exc}"))
    else:
        if model is None or not ids:
            problems.append(_err("index:bm25", "BM25 index is empty"))
        elif len(ids) != n_chunks:
            problems.append(
                _err(
                    "index:bm25-count",
                    f"BM25 covers {len(ids)} chunks but the index holds {n_chunks}",
                )
            )
    return problems


def stored_analyzer_version(conn: sqlite3.Connection) -> int | None:
    """``ANALYZER_VERSION`` recorded in the persisted BM25 blob, or None.

    None means there is no usable blob at all — no version to disagree with,
    and the separate "BM25 index is empty" check covers that case.
    """
    import pickle

    try:
        row = conn.execute("select data from bm25 where id = 1").fetchone()
    except sqlite3.Error:
        return None
    if not row or not row["data"]:
        return None
    try:
        return int(pickle.loads(row["data"]).get("analyzer_version", 1))
    except Exception:  # noqa: BLE001 - unreadable blob is not a version mismatch
        return None


# --- retrieval-level --------------------------------------------------------


def check_retrieval(
    conn: sqlite3.Connection,
    specs: list[TitleSpec],
    *,
    n_probe: int = DEFAULT_PROBE_SECTIONS,
    seed: int = PROBE_SEED,
) -> list[Problem]:
    """Functional probe: can a section be found by quoting its own citation?

    Every structural check above can pass on an index that retrieves nothing —
    a broken analyzer, a BM25 index built over the wrong column, a tokenizer
    change that stopped emitting citation tokens. This is the cheapest query
    whose right answer is known without a gold set: ask for ``4 CFR § 7.3``
    and the top section must be ``4 CFR § 7.3``.

    Sampling is seeded, so a failure is reproducible and the gate does not
    flicker between runs.
    """
    import random

    from retrieval.search import bm25_search

    problems: list[Problem] = []
    for spec in specs:
        sections = [
            r[0]
            for r in conn.execute(
                "select distinct section_citation from chunks where title = ? "
                "order by section_citation",
                (spec.number,),
            )
        ]
        if not sections:
            continue  # already reported by check_index
        rng = random.Random(f"{seed}:{spec.slug}")
        probe = rng.sample(sections, min(n_probe, len(sections)))
        misses = []
        for citation in probe:
            hits = bm25_search(citation, 1, conn=conn)
            if not hits or hits[0].section_citation != citation:
                got = hits[0].section_citation if hits else "nothing"
                misses.append(f"{citation} -> {got}")
        if misses:
            rate = len(misses) / len(probe)
            problem = _err if rate > 0.2 else _warn
            problems.append(
                problem(
                    "retrieval:self-lookup",
                    f"{spec.label}: {len(misses)}/{len(probe)} sampled sections are not "
                    f"rank 1 for their own citation (e.g. {misses[0]})",
                )
            )
    return problems


# --- parameters -------------------------------------------------------------


def check_params(specs: list[TitleSpec], gold_path: Path | None = None) -> list[Problem]:
    """Were the retrieval parameters tuned against *this* corpus?

    The fusion constants are calibrated to a score distribution that moves
    whenever the corpus does. When Title 1 was onboarded, nothing re-tuned
    them and hybrid's prose MRR fell below vector-only's with no stage
    noticing. This check is the mechanism that comment used to be: a warning
    that names what changed, cleared by running the tune stage (which records
    "checked, still best" when that is the outcome).
    """
    from eval import run as eval_run
    from retrieval import params as fusion_params
    from retrieval import tune

    _, prov = fusion_params.load()
    gold = eval_run.load_gold(gold_path or eval_run.DEFAULT_GOLD) if (
        gold_path or eval_run.DEFAULT_GOLD
    ).exists() else []
    from rag.reranker import DEFAULT_RERANK_MODEL

    reasons = tune.staleness(prov, specs, gold, DEFAULT_RERANK_MODEL)
    if not reasons:
        return []
    return [
        _warn(
            "params:stale",
            "retrieval parameters were not tuned for this corpus — " + "; ".join(reasons)
            + " — run `python -m retrieval.tune` (the onboard pipeline's tune stage)",
        )
    ]


# --- gate -------------------------------------------------------------------


def run_checks(
    specs: list[TitleSpec],
    *,
    db_path: Path = DEFAULT_DB,
    chunks_dir: Path | None = None,
    probe: bool = True,
    n_probe: int = DEFAULT_PROBE_SECTIONS,
    params: bool = True,
) -> list[Problem]:
    """Every check, over every declared title. Order is chunk → index → retrieval → params."""
    problems: list[Problem] = []
    counts: dict[int, int] = {}
    all_ids: dict[str, int] = {}

    for spec in specs:
        path = spec.chunks_path(chunks_dir) if chunks_dir else spec.chunks_path()
        if not path.exists():
            problems.append(
                _err("chunk:missing-file", f"{spec.label}: no chunk file at {path}")
            )
            continue
        chunks = _read_chunks(path)
        counts[spec.number] = len(chunks)
        problems.extend(check_chunk_file(spec, chunks))
        for c in chunks:
            cid = str(c.get("chunk_id"))
            if cid in all_ids and all_ids[cid] != spec.number:
                problems.append(
                    _err(
                        "chunk:cross-title-id",
                        f"chunk_id {cid} is used by both Title {all_ids[cid]} "
                        f"and {spec.label}; one would overwrite the other",
                    )
                )
            all_ids[cid] = spec.number

    if not Path(db_path).exists():
        problems.append(_err("index:missing", f"no index at {db_path}"))
        return problems

    conn = connect(db_path)
    try:
        problems.extend(check_index(conn, specs, counts))
        # A probe against a demonstrably broken index produces noise, not
        # information, so it only runs once the structure is sound.
        if probe and not any(p.severity == ERROR for p in problems):
            problems.extend(check_retrieval(conn, specs, n_probe=n_probe))
    finally:
        conn.close()
    if params:
        problems.extend(check_params(specs))
    return problems


def summarize(problems: list[Problem], *, strict: bool = False) -> tuple[bool, str]:
    """Return ``(ok, text)``. With ``strict``, warnings fail the gate too."""
    errors = [p for p in problems if p.severity == ERROR]
    warnings = [p for p in problems if p.severity == WARNING]
    lines = [str(p) for p in problems]
    ok = not errors and (not warnings or not strict)
    if not problems:
        lines.append("integrity: all checks passed")
    else:
        lines.append(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return ok, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="python -m pipeline.integrity",
        description="Gate the declared corpus: chunk, index and retrieval checks.",
    )
    parser.add_argument("--manifest", type=Path, default=manifest.DEFAULT_MANIFEST)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--no-probe", action="store_true", help="Skip the retrieval probe")
    parser.add_argument("--probe-sections", type=int, default=DEFAULT_PROBE_SECTIONS)
    args = parser.parse_args(argv)

    try:
        specs = manifest.load(args.manifest)
    except manifest.ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    problems = run_checks(
        specs,
        db_path=args.db,
        probe=not args.no_probe,
        n_probe=args.probe_sections,
    )
    ok, text = summarize(problems, strict=args.strict)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
