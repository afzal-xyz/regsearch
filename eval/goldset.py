"""Gold-set schema, coverage gate, and draft scaffolding for a new title.

The gold set is the only thing standing between "the pipeline ran" and "the
pipeline produced something good". Onboarding a title without extending it
means the new text is searched but never measured — the index grows, every
metric stays flat because it is averaged over questions that do not touch the
new title, and a retrieval regression there is invisible. So coverage is a
gate, not a suggestion.

What can and cannot be automated is a real line, and this module draws it
explicitly:

- **Citation lookups are generated.** "What does 4 CFR § 7.3 provide?" has a
  correct answer by construction — the section named is the section wanted.
- **Negatives are generated.** A citation-form negative names a section number
  verified absent from the corpus, so its empty answer is correct by
  construction too.
- **Prose questions are not generated.** The whole point of a prose question
  is that it asks in everyday words a regulation does not use — "can I bring a
  gun in" against text that says "firearms". A question generated *from* the
  section body necessarily reuses that body's vocabulary, which makes it a
  test of string matching rather than retrieval, and it would score
  near-perfectly while measuring nothing. So this module emits prose *stubs*:
  a stratified sample of sections with their text, for a human to write
  against. :func:`validate` rejects an unanswered stub, so a stub can never
  drift into the measured set.

CLI:
    python -m eval.goldset --check                      # validate + coverage gate
    python -m eval.goldset --draft 1 --n-prose 14       # scaffold a new title
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from index.store import DEFAULT_DB, connect
from pipeline import manifest
from pipeline.manifest import TitleSpec

logger = logging.getLogger(__name__)

DEFAULT_GOLD = Path("eval/gold.jsonl")
DEFAULT_DRAFT_DIR = Path("eval")

KINDS = ("prose", "citation")
STATUSES = ("approved", "draft")

# Minimum approved questions per declared title before the corpus is
# considered measured. These are a floor, not a target: the established
# titles carry 42 prose + 12 citation each.
#
# 12 is where the floor sits because the harness reports per-kind metrics and
# compare.py gates regressions at 1.5/n. Below ~12 questions a single flipped
# question moves recall by more than 8 points, so the per-title numbers are
# too noisy to tell a real regression from a coin flip — the gate would either
# never fire or fire constantly.
MIN_PROSE_PER_TITLE = 12
MIN_CITATION_PER_TITLE = 4
# Negatives are corpus-wide rather than per-title (a negative belongs to no
# title — that is the point), but each new title brings new ways to be wrongly
# confident, so the required count scales with the corpus.
MIN_NEGATIVES_PER_TITLE = 2

DRAFT_SEED = 20260730


@dataclass
class Problem:
    check: str
    severity: str  # "error" | "warning"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity:<7}] {self.check}: {self.message}"


def _err(check: str, message: str) -> Problem:
    return Problem(check, "error", message)


def _warn(check: str, message: str) -> Problem:
    return Problem(check, "warning", message)


# --- loading ----------------------------------------------------------------


def load(path: Path | str = DEFAULT_GOLD) -> list[dict]:
    """Every entry in the gold file, approved or not."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n} is not valid JSON: {exc}") from exc
    return out


def write(entries: list[dict], path: Path | str) -> Path:
    """Write entries as JSONL, one compact object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


def titles_of(entry: dict) -> set[int]:
    """CFR title numbers a gold entry's citations refer to."""
    out = set()
    for c in entry.get("relevant_citations") or []:
        head = str(c).split(" CFR", 1)[0].strip()
        if head.isdigit():
            out.add(int(head))
    return out


def is_negative(entry: dict) -> bool:
    return not entry.get("relevant_citations")


# --- validation -------------------------------------------------------------


def validate(
    entries: list[dict],
    *,
    sections: set[str] | None = None,
    specs: list[TitleSpec] | None = None,
) -> list[Problem]:
    """Schema and consistency checks over the whole gold file.

    ``sections`` is every section citation in the index; when given, each
    cited section must exist (a typo'd citation is otherwise an unanswerable
    question that silently drags recall down forever). ``specs`` is the
    declared corpus, used to flag negatives that predate it.
    """
    problems: list[Problem] = []
    seen: dict[str, int] = {}

    for i, e in enumerate(entries, start=1):
        where = f"entry {i}"
        question = str(e.get("question", "")).strip()
        if not question:
            problems.append(
                _err("gold:empty-question", f"{where} has no question (an unfilled draft stub?)")
            )
        elif question in seen:
            problems.append(
                _err(
                    "gold:duplicate",
                    f"{where} repeats the question at entry {seen[question]}: {question[:60]!r} "
                    f"— duplicates silently double-weight one question in the metrics",
                )
            )
        else:
            seen[question] = i

        status = e.get("status", "approved")
        if status not in STATUSES:
            problems.append(
                _err("gold:status", f"{where} has status {status!r}, expected one of {STATUSES}")
            )
        kind = e.get("kind", "prose")
        if kind not in KINDS:
            problems.append(
                _err("gold:kind", f"{where} has kind {kind!r}, expected one of {KINDS}")
            )

        cites = e.get("relevant_citations")
        if cites is None or not isinstance(cites, list):
            problems.append(
                _err(
                    "gold:citations",
                    f"{where} needs a `relevant_citations` list (use [] for a negative)",
                )
            )
            continue
        if sections is not None:
            for c in cites:
                if c not in sections:
                    problems.append(
                        _err(
                            "gold:unknown-citation",
                            f"{where} cites {c!r}, which is not a section in the index "
                            f"— the question can never be answered correctly",
                        )
                    )

    if specs is not None:
        problems.extend(_check_negatives(entries, specs, sections))
    return problems


def _check_negatives(
    entries: list[dict], specs: list[TitleSpec], sections: set[str] | None
) -> list[Problem]:
    """Negatives are only negative with respect to a particular corpus.

    "What is the federal income tax filing deadline?" has no answer in Titles
    4 and 31, and a correct answer the moment Title 26 is onboarded. Nothing
    about the entry changes — it just quietly becomes a question the system is
    penalised for answering *correctly*. So each negative records the corpus
    it was verified against, and growing the corpus past that marks it stale.
    """
    problems: list[Problem] = []
    current = set(manifest.corpus_state(specs))
    negatives = [e for e in entries if is_negative(e)]

    stale = []
    unmarked = []
    for e in negatives:
        verified = e.get("negative_corpus")
        if verified is None:
            unmarked.append(e)
            continue
        missing = current - set(verified)
        if missing:
            stale.append((e, sorted(missing)))

    if unmarked:
        problems.append(
            _warn(
                "gold:negative-unverified",
                f"{len(unmarked)} negative(s) have no `negative_corpus` field, so there is no "
                f"record of which corpus they were checked against "
                f"(e.g. {unmarked[0].get('question', '')[:50]!r})",
            )
        )
    if stale:
        added = sorted({t for _, miss in stale for t in miss})
        problems.append(
            _warn(
                "gold:negative-stale",
                f"{len(stale)} negative(s) were verified before {added} joined the corpus; "
                f"re-read them and confirm the new text still does not answer them, then "
                f"update `negative_corpus` "
                f"(e.g. {stale[0][0].get('question', '')[:50]!r})",
            )
        )

    # A citation-form negative names a section on purpose; if that section is
    # now indexed, the entry is simply wrong rather than merely stale.
    if sections is not None:
        import re

        for e in negatives:
            question = str(e.get("question", ""))
            for m in re.finditer(r"(\d+)\s+CFR\s+§?\s*([\d.]+[\w().-]*)", question):
                cited = f"{m.group(1)} CFR § {m.group(2).rstrip('.')}"
                if cited in sections:
                    problems.append(
                        _err(
                            "gold:negative-answerable",
                            f"negative {e.get('question', '')[:50]!r} cites {cited}, which IS in "
                            f"the index — it is no longer a negative",
                        )
                    )
    return problems


# --- coverage ---------------------------------------------------------------


@dataclass
class Coverage:
    per_title: dict[int, dict[str, int]] = field(default_factory=dict)
    negatives: int = 0


def coverage(entries: list[dict], specs: list[TitleSpec]) -> Coverage:
    """Approved question counts per declared title, by kind, plus negatives."""
    cov = Coverage(per_title={s.number: dict.fromkeys(KINDS, 0) for s in specs})
    for e in entries:
        if e.get("status", "approved") != "approved":
            continue
        if is_negative(e):
            cov.negatives += 1
            continue
        kind = e.get("kind", "prose")
        if kind not in KINDS:
            continue
        for t in titles_of(e):
            cov.per_title.setdefault(t, dict.fromkeys(KINDS, 0))
            cov.per_title[t][kind] += 1
    return cov


def check_coverage(
    cov: Coverage,
    specs: list[TitleSpec],
    *,
    min_prose: int = MIN_PROSE_PER_TITLE,
    min_citation: int = MIN_CITATION_PER_TITLE,
    min_negatives_per_title: int = MIN_NEGATIVES_PER_TITLE,
) -> list[Problem]:
    """Gate: is every declared title measured well enough to detect a regression?"""
    problems: list[Problem] = []
    for spec in specs:
        counts = cov.per_title.get(spec.number, dict.fromkeys(KINDS, 0))
        if counts["prose"] < min_prose:
            problems.append(
                _err(
                    "gold:coverage-prose",
                    f"{spec.label}: {counts['prose']} approved prose questions, need {min_prose} "
                    f"— scaffold with `python -m eval.goldset --draft {spec.number}`",
                )
            )
        if counts["citation"] < min_citation:
            problems.append(
                _err(
                    "gold:coverage-citation",
                    f"{spec.label}: {counts['citation']} approved citation lookups, "
                    f"need {min_citation}",
                )
            )
    required = min_negatives_per_title * len(specs)
    if cov.negatives < required:
        problems.append(
            _err(
                "gold:coverage-negatives",
                f"{cov.negatives} approved negatives, need {required} "
                f"({min_negatives_per_title} per declared title)",
            )
        )
    return problems


# --- draft scaffolding ------------------------------------------------------

_CITATION_TEMPLATES = (
    "What does {cite} provide?",
    "What are the requirements in {cite}?",
    "Summarize {cite}.",
    "What does {cite} say?",
)


def _load_title_sections(conn, title: int) -> list[dict]:
    """One row per section of ``title``: citation, heading, and joined body."""
    rows = conn.execute(
        "select section_citation, heading, text, char_start from chunks "
        "where title = ? order by section_citation, char_start",
        (title,),
    ).fetchall()
    by_section: dict[str, dict] = {}
    for r in rows:
        sec = by_section.setdefault(
            r["section_citation"],
            {"section_citation": r["section_citation"], "heading": r["heading"], "pieces": []},
        )
        sec["pieces"].append(r["text"])
    out = []
    for sec in by_section.values():
        sec["body"] = "\n\n".join(sec["pieces"])
        del sec["pieces"]
        out.append(sec)
    return sorted(out, key=lambda s: s["section_citation"])


def _part_of(section_citation: str) -> str:
    import re

    m = re.search(r"§\s*([\w]+)", section_citation)
    return m.group(1).split(".")[0] if m else "?"


def _stratified(sections: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample ``n`` sections spread across CFR parts, not clustered in the big one.

    A uniform sample of a title whose Part 1010 holds half the sections puts
    half the questions there, and the small parts — often the ones with
    distinctive vocabulary — go unmeasured.
    """
    by_part: dict[str, list[dict]] = {}
    for s in sections:
        by_part.setdefault(_part_of(s["section_citation"]), []).append(s)
    parts = sorted(by_part)
    picked: list[dict] = []
    # Round-robin across parts, so every part contributes before any contributes twice.
    pools = {p: rng.sample(by_part[p], len(by_part[p])) for p in parts}
    while len(picked) < n and any(pools.values()):
        for p in parts:
            if not pools[p]:
                continue
            picked.append(pools[p].pop())
            if len(picked) >= n:
                break
    return sorted(picked, key=lambda s: s["section_citation"])


def draft_citation_questions(
    sections: list[dict], n: int, rng: random.Random, corpus: list[str]
) -> list[dict]:
    """Known-item lookups. Correct by construction: the section named is the answer."""
    picked = _stratified(sections, n, rng)
    out = []
    for i, s in enumerate(picked):
        cite = s["section_citation"]
        out.append(
            {
                "question": _CITATION_TEMPLATES[i % len(_CITATION_TEMPLATES)].format(cite=cite),
                "relevant_citations": [cite],
                "kind": "citation",
                "notes": f"Auto-generated known-item lookup for {cite} ({s['heading']!r}).",
                "status": "draft",
                "generated": True,
            }
        )
    return out


def draft_negatives(
    title: int, sections: list[dict], n: int, rng: random.Random, corpus: list[str]
) -> list[dict]:
    """Citation-form negatives naming a section number verified absent."""
    present = {s["section_citation"] for s in sections}
    parts = sorted({p for p in (_part_of(s["section_citation"]) for s in sections) if p != "?"})
    out: list[dict] = []
    for part in rng.sample(parts, min(len(parts), n)):
        cite = f"{title} CFR § {part}.9999"
        if cite in present:  # pragma: no cover - .9999 is not a real section number
            continue
        out.append(
            {
                "question": f"What are the recordkeeping requirements in {title} CFR {part}.9999?",
                "relevant_citations": [],
                "kind": "citation",
                "notes": (
                    f"Auto-generated negative: § {part}.9999 does not exist in Part {part}. "
                    f"Verifies that a citation to a non-existent section is not forced into "
                    f"a confident answer."
                ),
                "status": "draft",
                "generated": True,
                "negative_corpus": corpus,
            }
        )
    return out


def draft_prose_stubs(
    sections: list[dict], n: int, rng: random.Random, excerpt: int = 600
) -> list[dict]:
    """Unanswered stubs for a human to write prose questions against.

    Each carries the section's heading and the opening of its body, which is
    what the author needs in order to ask about it in words the text does not
    use. ``question`` is deliberately empty: :func:`validate` errors on an
    empty question, so a stub cannot be mistaken for a finished entry.
    """
    out = []
    for s in _stratified(sections, n, rng):
        body = " ".join(s["body"].split())
        out.append(
            {
                "question": "",
                "relevant_citations": [s["section_citation"]],
                "kind": "prose",
                "notes": (
                    f"TODO write a question. Source {s['section_citation']} "
                    f"({s['heading']!r}). Ask in everyday words the text does NOT use, "
                    f"so a hit needs meaning rather than string overlap. "
                    f"Body: {body[:excerpt]}{'…' if len(body) > excerpt else ''}"
                ),
                "status": "draft",
                "generated": True,
            }
        )
    return out


def draft_for_title(
    spec: TitleSpec,
    *,
    db_path: Path = DEFAULT_DB,
    n_prose: int = MIN_PROSE_PER_TITLE,
    n_citation: int = MIN_CITATION_PER_TITLE,
    n_negatives: int = MIN_NEGATIVES_PER_TITLE,
    corpus: list[str] | None = None,
    seed: int = DRAFT_SEED,
) -> list[dict]:
    """Full draft gold set for one title: citation lookups, negatives, prose stubs."""
    conn = connect(db_path)
    try:
        sections = _load_title_sections(conn, spec.number)
    finally:
        conn.close()
    if not sections:
        raise ValueError(f"{spec.label} has no sections in {db_path}; index it first")

    corpus = corpus or [spec.slug]
    # One seeded RNG per category, so changing the count of one does not
    # reshuffle the others and churn an already-reviewed draft file.
    return [
        *draft_citation_questions(
            sections, n_citation, random.Random(f"{seed}:cite:{spec.slug}"), corpus
        ),
        *draft_negatives(
            spec.number, sections, n_negatives, random.Random(f"{seed}:neg:{spec.slug}"), corpus
        ),
        *draft_prose_stubs(sections, n_prose, random.Random(f"{seed}:prose:{spec.slug}")),
    ]


# --- gate -------------------------------------------------------------------


def index_sections(db_path: Path = DEFAULT_DB) -> set[str]:
    if not Path(db_path).exists():
        return set()
    conn = connect(db_path)
    try:
        return {r[0] for r in conn.execute("select distinct section_citation from chunks")}
    finally:
        conn.close()


def run_checks(
    specs: list[TitleSpec],
    *,
    gold_path: Path = DEFAULT_GOLD,
    db_path: Path = DEFAULT_DB,
    min_prose: int = MIN_PROSE_PER_TITLE,
    min_citation: int = MIN_CITATION_PER_TITLE,
) -> list[Problem]:
    entries = load(gold_path)
    if not entries:
        return [_err("gold:missing", f"no gold entries in {gold_path}")]
    sections = index_sections(db_path) or None
    problems = validate(entries, sections=sections, specs=specs)
    problems.extend(
        check_coverage(
            coverage(entries, specs), specs, min_prose=min_prose, min_citation=min_citation
        )
    )
    return problems


def _print_coverage(cov: Coverage, specs: list[TitleSpec]) -> None:
    print(f"\n{'title':<8} {'prose':>7} {'citation':>9}")
    print("-" * 26)
    for spec in specs:
        c = cov.per_title.get(spec.number, dict.fromkeys(KINDS, 0))
        print(f"{spec.number:<8} {c['prose']:>7} {c['citation']:>9}")
    print(f"{'(neg)':<8} {cov.negatives:>7}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="python -m eval.goldset",
        description="Validate the gold set, gate its coverage, or scaffold a new title.",
    )
    parser.add_argument("--manifest", type=Path, default=manifest.DEFAULT_MANIFEST)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--check", action="store_true", help="Validate and gate coverage")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--draft", type=int, metavar="TITLE", help="Scaffold a draft gold set")
    parser.add_argument("--n-prose", type=int, default=MIN_PROSE_PER_TITLE)
    parser.add_argument("--n-citation", type=int, default=MIN_CITATION_PER_TITLE)
    parser.add_argument("--n-negatives", type=int, default=MIN_NEGATIVES_PER_TITLE)
    parser.add_argument("--out", type=Path, help="Draft output path")
    args = parser.parse_args(argv)

    try:
        specs = manifest.load(args.manifest)
    except manifest.ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.draft is not None:
        spec = next((s for s in specs if s.number == args.draft), None)
        if spec is None:
            print(
                f"error: Title {args.draft} is not in {args.manifest}; add it first",
                file=sys.stderr,
            )
            return 1
        try:
            entries = draft_for_title(
                spec,
                db_path=args.db,
                n_prose=args.n_prose,
                n_citation=args.n_citation,
                n_negatives=args.n_negatives,
                corpus=manifest.corpus_state(specs),
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        out = args.out or DEFAULT_DRAFT_DIR / f"gold_title{spec.number}_draft.jsonl"
        write(entries, out)
        n_stub = sum(1 for e in entries if not e["question"])
        print(f"wrote {len(entries)} draft entries -> {out}")
        print(
            f"  {len(entries) - n_stub} generated and answerable as-is "
            f"(citation lookups + negatives)\n"
            f"  {n_stub} prose stubs need a question written by hand\n"
            f"Review, set \"status\": \"approved\", then merge into {args.gold}."
        )
        return 0

    entries = load(args.gold)
    problems = run_checks(specs, gold_path=args.gold, db_path=args.db)
    for p in problems:
        print(p)
    _print_coverage(coverage(entries, specs), specs)

    errors = [p for p in problems if p.severity == "error"]
    warnings = [p for p in problems if p.severity == "warning"]
    if not problems:
        print("\ngold set: all checks passed")
    else:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return 0 if not errors and (not warnings or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
