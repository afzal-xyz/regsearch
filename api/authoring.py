"""Gold-question authoring: draft queues, the lexical overlap meter, promotion.

The job this supports is writing a question that a retriever must *understand*
to answer. The failure mode is subtle: an author reads ``§ 2.6 Veterans'
preference``, writes "What is the veterans' preference policy?", and has
produced a question BM25 answers by string overlap alone. It will score well
forever and measure nothing. The approved set's own notes are full of the
cure — "De-vocabbed: 'served in the military' instead of 'veterans'
preference'" — but it has been a habit, not a mechanism.

So this module gives the habit a number.

Why an overlap meter and not a search box
-----------------------------------------
The obvious tool is a "test this question" button. It is the wrong tool, and
the reason is worth stating precisely, because the two look similar.

A search preview reports **whether the author won**. Iterating against it
selects for questions the current engine already answers, which is precisely
the correlation a benchmark exists to measure. The gold set stops being an
independent yardstick and becomes a record of the engine's current shape.

The overlap meter reports **how much of the section's distinctive vocabulary
the question reuses**. It is one-sided: driving it green can only remove
lexical shortcuts, making the question strictly harder for BM25. There is no
setting of the meter that tells you retrieval succeeded, so there is nothing
to overfit *to*. That asymmetry is the whole argument, and it is why the meter
may reuse the BM25 analyzer — measuring overlap in the exact vocabulary the
retriever sees is the point — while a search box may not exist at all.

One honest caveat, recorded so it is not discovered later: a meter pushed to
zero everywhere would bias the set toward vector-friendly paraphrase and away
from legitimately keyword-answerable questions, which are also real. Amber is
the normal target. The meter warns; it never blocks a save.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import threading
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from eval import goldset
from eval import run as eval_run
from eval.goldset import Problem
from index.store import DEFAULT_DB, connect
from index.tokenizer import citation_tokens, tokenize
from pipeline import manifest
from pipeline.manifest import TitleSpec

logger = logging.getLogger(__name__)

DEFAULT_EVAL_DIR = Path("eval")
DRAFT_TEMPLATE = "gold_title{title}_draft.jsonl"

# Fields the authoring UI keeps on a draft entry but that must never reach
# eval/gold.jsonl: they describe the authoring session, not the question.
DRAFT_ONLY_FIELDS = ("draft_id", "generated", "revealed", "overlap")

# --- meter thresholds -------------------------------------------------------
#
# CALIBRATED, not guessed. Provenance, in the style of retrieval/params.json:
#
#   corpus        1@2024-07-18, 4@2024-07-18, 31@2024-07-18  (5,849 sections)
#   gold          d94a9cf8499ed414 — 98 approved prose questions, all scorable
#   calibrated    2026-11-07, via `python -m api.authoring --calibrate`
#
# Distinctive-overlap distribution over those 98 already-approved questions:
#
#   p10 0.160   p25 0.283   p50 0.397   p75 0.548
#   p90 0.676   p95 0.757   p99 0.849   p100 0.884
#
# The first guesses (amber 0.25, red 0.45) were far too tight: they painted
# 42% of the approved set amber and 37% of it red. A warning that fires on a
# third of human-reviewed work is not a warning, it is a background colour,
# and the first thing an author learns is to ignore it. The guesses were wrong
# in an understandable way — a question genuinely about one section shares a
# lot of moderately-rare vocabulary with it, unavoidably, and the real median
# is 0.40 rather than the ~0.15 the bands implied.
#
# Amber now sits at p75 and red at p95 of the approved distribution: the
# quartile worth a second look, and the tail that really is derivative (the
# calibrate listing names them, and they are worth rewriting). Measured
# effect on the approved set: 19% amber, 5% red.
#
# Re-run the calibration whenever the corpus or the gold set changes; the
# distribution is a property of both.
DISTINCTIVE_AMBER = 0.55  # p75
DISTINCTIVE_RED = 0.76  # p95

# Rarity above which a shared term is a term of art rather than a common word.
# On the calibrated corpus 0.80 means "appears in at most ~5 of 5,849
# sections" (rarity 0.814 at df=5, 0.735 at df=10) — which is the right shape
# for "term of art". Measured: 97% of approved questions copy none of these,
# 3% copy exactly one, and none copies two. So the rule below is a tripwire
# that has never yet fired on reviewed work, which is what a tripwire should
# be; it is not doing the everyday warning, the band above is.
RARE_TERM = 0.80
RARE_COPIED_RED = 2

GREEN, AMBER, RED = "green", "amber", "red"
# Not verdicts. These are the states where the meter has nothing to say, and
# they are deliberately not GREEN.
#
# "I checked and the question borrows nothing" and "I could not check" are
# opposite facts that a traffic light renders identically, and the second one
# shown as green is the dangerous direction: an author scanning a queue for
# green badges would approve a question nobody verified. So an unindexed
# title, a section missing from `chunks`, or a section whose body is empty all
# land on UNKNOWN, which reads "not checked" and blocks approval.
EMPTY = "empty"  # no question written yet
UNKNOWN = "unknown"  # could not check — never treat as clean
NOT_APPLICABLE = "n/a"  # a negative: there is no section to borrow from

# States in which the meter has made no judgement about the question.
UNJUDGED = (EMPTY, UNKNOWN, NOT_APPLICABLE)


# --- corpus vocabulary ------------------------------------------------------


@dataclass(frozen=True)
class Vocabulary:
    """Section-level document frequencies for the whole indexed corpus.

    Section-level, not chunk-level: the author is distinguishing one section
    from the rest, so "how many sections use this word" is the quantity that
    says whether a word is distinctive here.
    """

    n_sections: int
    df: dict[str, int] = field(default_factory=dict)

    def rarity(self, token: str) -> float:
        """0 for a word in every section, 1 for a word in exactly one.

        A token the corpus has never seen scores 1: the author brought it, and
        bringing your own words is the behaviour being encouraged.
        """
        if self.n_sections <= 1:
            return 1.0
        return self.rarity_for_df(self.df.get(token, 0))

    def rarity_for_df(self, df: int) -> float:
        """The rarity a term with this document frequency would score."""
        if self.n_sections <= 1 or df <= 0:
            return 1.0
        return min(1.0, math.log(self.n_sections / df) / math.log(self.n_sections))


EMPTY_VOCABULARY = Vocabulary(n_sections=0)

_VOCAB_CACHE: dict[tuple, Vocabulary] = {}
_VOCAB_LOCK = threading.Lock()


def build_vocabulary(conn) -> Vocabulary:
    """Tokenize every section once and count how many contain each term."""
    bodies: dict[str, list[str]] = {}
    for row in conn.execute(
        "select section_citation, heading, text from chunks order by section_citation"
    ):
        bodies.setdefault(row["section_citation"], []).append(
            f"{row['heading'] or ''}\n{row['text'] or ''}"
        )
    df: Counter[str] = Counter()
    for pieces in bodies.values():
        df.update(set(_content_tokens("\n".join(pieces))))
    return Vocabulary(n_sections=len(bodies), df=dict(df))


def vocabulary(db_path: Path | str = DEFAULT_DB) -> Vocabulary:
    """Cached corpus vocabulary, rebuilt when the index file changes.

    Built lazily on the first authoring request and then reused. It is a SQL
    scan plus the BM25 analyzer — no model, no embedding, no retrieval — but
    it is seconds of work on a real corpus, so callers should stay off the
    event loop (the routes are sync, so Starlette runs them in a threadpool).
    """
    path = Path(db_path)
    if not path.exists():
        return EMPTY_VOCABULARY
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    with _VOCAB_LOCK:
        cached = _VOCAB_CACHE.get(key)
        if cached is not None:
            return cached
    conn = connect(path)
    try:
        built = build_vocabulary(conn)
    finally:
        conn.close()
    with _VOCAB_LOCK:
        _VOCAB_CACHE.clear()  # only the current index is worth keeping
        _VOCAB_CACHE[key] = built
    return built


def reset_vocabulary_cache() -> None:
    with _VOCAB_LOCK:
        _VOCAB_CACHE.clear()


# --- the meter --------------------------------------------------------------


def _content_tokens(text: str) -> list[str]:
    """Non-citation BM25 tokens: stopwords dropped, plurals folded.

    Citations are excluded because they are handled separately and far more
    harshly — a prose question naming its own section is not a vocabulary
    problem, it is a different kind of question.
    """
    cites = citation_tokens(text)
    return [t for t in tokenize(text) if t not in cites]


@dataclass(frozen=True)
class SharedTerm:
    term: str
    df: int
    rarity: float

    @property
    def distinctive(self) -> bool:
        return self.rarity >= RARE_TERM


@dataclass(frozen=True)
class Overlap:
    """What a question reuses from the section it is about."""

    verdict: str
    distinctive: float  # rarity-weighted share of the question's vocabulary
    jaccard: float  # plain set overlap, for reference
    shared: tuple[SharedTerm, ...]  # rarest first — the actionable part
    question_terms: int
    names_citation: bool
    advice: str

    @property
    def rare_copied(self) -> tuple[str, ...]:
        return tuple(s.term for s in self.shared if s.distinctive)

    @property
    def percent(self) -> int:
        return round(self.distinctive * 100)

    @property
    def checked(self) -> bool:
        """Did the meter actually compare the question against a section body?"""
        return self.verdict not in UNJUDGED

    @property
    def unverified(self) -> bool:
        """Could not check, and it matters — distinct from "nothing to check"."""
        return self.verdict == UNKNOWN

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            # Explicit, so a JSON consumer cannot read distinctive=0.0 as
            # "clean" when the truth is "never compared against anything".
            "checked": self.checked,
            "unverified": self.unverified,
            "distinctive": round(self.distinctive, 4) if self.checked else None,
            "jaccard": round(self.jaccard, 4) if self.checked else None,
            "shared": [{"term": s.term, "df": s.df, "rarity": round(s.rarity, 3)}
                       for s in self.shared],
            "names_citation": self.names_citation,
            "advice": self.advice,
        }


def not_applicable(reason: str) -> Overlap:
    """A negative question: there is no section, so there is nothing to borrow."""
    return Overlap(NOT_APPLICABLE, 0.0, 0.0, (), 0, False, reason)


def overlap(
    question: str,
    body: str,
    *,
    vocab: Vocabulary | None = None,
    kind: str = "prose",
    unchecked_reason: str = "",
) -> Overlap:
    """Score how much distinctive vocabulary ``question`` reuses from ``body``.

    ``distinctive`` is the rarity-weighted fraction of the question's content
    vocabulary that also appears in the section: 0 when the author shares
    nothing but common words, 1 when every word they chose came from the text.
    Weighting by rarity is what separates the two failure modes — sharing
    "employee" with a section about employees is unavoidable and harmless;
    sharing "preference eligible" is the author copying a term of art.
    """
    vocab = vocab or EMPTY_VOCABULARY
    q_tokens = _content_tokens(question)
    if not q_tokens:
        return Overlap(EMPTY, 0.0, 0.0, (), 0, False, "Write a question to see the meter.")

    q_set = set(q_tokens)
    body_set = set(_content_tokens(body))
    if not body_set:
        return Overlap(
            UNKNOWN, 0.0, 0.0, (), len(q_set), False,
            unchecked_reason
            or "There is no section text to compare against, so this question has NOT been "
            "checked. Do not read this as clean.",
        )
    shared_terms = q_set & body_set

    weights = {t: vocab.rarity(t) for t in q_set}
    total = sum(weights.values())
    copied = sum(weights[t] for t in shared_terms)
    distinctive = (copied / total) if total else 0.0

    union = q_set | body_set
    jaccard = len(shared_terms) / len(union) if union else 0.0

    shared = tuple(
        sorted(
            (SharedTerm(t, vocab.df.get(t, 0), weights[t]) for t in shared_terms),
            key=lambda s: (-s.rarity, s.term),
        )
    )
    # A prose question that names its own section is answerable by the
    # citation-aware tokenizer alone; it is a citation lookup mislabelled.
    names_citation = kind == "prose" and bool(citation_tokens(question))

    rare = [s for s in shared if s.distinctive]
    if names_citation:
        verdict = RED
    elif distinctive >= DISTINCTIVE_RED or len(rare) >= RARE_COPIED_RED:
        verdict = RED
    elif distinctive >= DISTINCTIVE_AMBER or rare:
        verdict = AMBER
    else:
        verdict = GREEN

    return Overlap(
        verdict=verdict,
        distinctive=distinctive,
        jaccard=jaccard,
        shared=shared,
        question_terms=len(q_set),
        names_citation=names_citation,
        advice=_advice(verdict, names_citation, rare),
    )


def _advice(verdict: str, names_citation: bool, rare: list[SharedTerm]) -> str:
    if names_citation:
        return (
            "This prose question names a section number. The citation-aware tokenizer will "
            "match it directly, so it measures nothing about understanding — either rephrase "
            "without the citation, or file it as a citation lookup."
        )
    if verdict == RED and rare:
        terms = ", ".join(f"“{s.term}”" for s in rare[:3])
        return f"Replace the statutory wording — {terms} — with how a member of the public "\
               f"would say it."
    if verdict == RED:
        return (
            "Most of this question's vocabulary comes from the section text. Reread the "
            "heading, then ask the question from memory in your own words."
        )
    if verdict == AMBER:
        if rare:
            return f"“{rare[0].term}” is a term of art lifted from the text; try a plainer word."
        return "Some wording is shared with the section. Usually fine — check the terms below."
    return "Little distinctive vocabulary shared. A hit will need meaning, not string overlap."


# --- sections ---------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    citation: str
    heading: str
    breadcrumb: tuple[str, ...]
    body: str
    title: int | None = None

    @property
    def found(self) -> bool:
        return bool(self.body or self.heading)


MISSING_SECTION = Section("", "", (), "")


def load_section(conn, citation: str) -> Section:
    """Citation, heading, hierarchy breadcrumb and joined body for one section."""
    rows = conn.execute(
        "select heading, heading_path, title, text from chunks "
        "where section_citation = ? order by char_start",
        (citation,),
    ).fetchall()
    if not rows:
        return MISSING_SECTION
    path = rows[0]["heading_path"]
    if isinstance(path, str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            path = json.loads(path)
    crumbs = tuple(str(p) for p in (path or []))
    return Section(
        citation=citation,
        heading=rows[0]["heading"] or "",
        # Drop the leaf: it is the heading, already shown on its own.
        breadcrumb=crumbs[:-1] if crumbs and crumbs[-1] == (rows[0]["heading"] or "") else crumbs,
        body="\n\n".join(r["text"] or "" for r in rows),
        title=rows[0]["title"],
    )


def load_sections(db_path: Path | str, citations: Iterable[str]) -> dict[str, Section]:
    path = Path(db_path)
    wanted = [c for c in dict.fromkeys(citations) if c]
    if not path.exists() or not wanted:
        return {}
    conn = connect(path)
    try:
        return {c: load_section(conn, c) for c in wanted}
    finally:
        conn.close()


# --- draft storage ----------------------------------------------------------


def draft_path(title: int, eval_dir: Path | str = DEFAULT_EVAL_DIR) -> Path:
    return Path(eval_dir) / DRAFT_TEMPLATE.format(title=int(title))


def _atomic_write(path: Path, data: str | bytes) -> Path:
    """Write via a temp file in the same directory, then rename.

    ``os.replace`` is atomic within a filesystem, so a reader (or a crash)
    sees either the old file or the new one — never a half-written gold set.
    ``bytes`` are written verbatim, which is how a rollback restores a file
    to exactly what it was rather than to a re-serialisation of it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        if isinstance(data, bytes):
            fh = open(tmp, "wb")
        else:
            fh = open(tmp, "w", encoding="utf-8")
        with fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()
    return path


def _snapshot(path: Path) -> bytes | None:
    """The file's exact bytes, or ``None`` if it does not exist."""
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        return None


def _restore(path: Path, original: bytes | None) -> None:
    """Put ``path`` back to ``original`` (``None`` = the file did not exist).

    A no-op when the file already holds those bytes, so rolling back a write
    that never landed costs nothing and cannot itself fail.
    """
    if _snapshot(path) == original:
        return
    if original is None:
        Path(path).unlink(missing_ok=True)
    else:
        _atomic_write(Path(path), original)


def _as_jsonl(entries: Iterable[dict]) -> str:
    return "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)


# Held by every draft write and for the whole of a promotion, so a row saved
# from another tab cannot land between promotion reading the draft and
# rewriting it. Re-entrant because ``set_status`` saves while holding it.
# Other processes and text editors take no lock; promotion covers those by
# re-checking the draft's bytes immediately before it rewrites the file.
_DRAFT_LOCK = threading.RLock()


def save_draft(title: int, entries: list[dict], eval_dir: Path | str = DEFAULT_EVAL_DIR) -> Path:
    with _DRAFT_LOCK:
        return _atomic_write(draft_path(title, eval_dir), _as_jsonl(entries))


@dataclass(frozen=True)
class DraftFile:
    """A draft queue read from disk: usable entries, plus what was wrong with it."""

    path: Path
    entries: list[dict]
    problems: tuple[Problem, ...] = ()
    lines: int = 0

    @property
    def skipped(self) -> int:
        return self.lines - len(self.entries)


def read_draft(title: int, eval_dir: Path | str = DEFAULT_EVAL_DIR) -> DraftFile:
    """Read a draft queue, tolerating anything a hand-edited file can contain.

    These files are meant to be edited by hand — that is the point of JSONL —
    so they arrive with fields missing, fields of the wrong type, and the
    occasional line that is not JSON at all. None of that should be a 500:
    the whole queue would become unreachable because of one bad line, with no
    way to find out which one from the UI.

    So reading is total. A line that cannot be parsed is reported and skipped;
    an entry whose shape is wrong is repaired **in memory** and reported.

    Nothing is written back. Normalising on load and saving the result would
    rewrite a file the author is editing in a text editor, turn a GET into a
    mutation, and churn the git diff of a tracked file for anyone who merely
    opened the page. The repaired form reaches disk only when the author
    saves a row, which is an edit they asked for.
    """
    path = draft_path(title, eval_dir)
    if not path.exists():
        return DraftFile(path=path, entries=[])

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return DraftFile(
            path=path, entries=[],
            problems=(_problem("draft:unreadable", f"cannot read {path}: {exc}"),),
        )
    return _parse_draft(path, text)


def _parse_draft(path: Path, text: str) -> DraftFile:
    """Parse draft JSONL text. Total: see :func:`read_draft`."""
    entries: list[dict] = []
    problems: list[Problem] = []
    lines = 0
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        lines += 1
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(
                _problem("draft:bad-json", f"line {n} is not valid JSON and was skipped: {exc}")
            )
            continue
        if not isinstance(entry, dict):
            problems.append(
                _problem(
                    "draft:not-an-object",
                    f"line {n} is a {type(entry).__name__}, not an object, and was skipped",
                )
            )
            continue
        problems.extend(_normalise(entry, n))
        if not entry.get("draft_id"):
            entry["draft_id"] = _derived_id(len(entries), entry)
        entries.append(entry)

    return DraftFile(path=path, entries=entries, problems=tuple(problems), lines=lines)


def load_draft(title: int, eval_dir: Path | str = DEFAULT_EVAL_DIR) -> list[dict]:
    """The usable entries of a draft queue. See :func:`read_draft`."""
    return read_draft(title, eval_dir).entries


def _derived_id(index: int, entry: dict) -> str:
    """A stable handle for an entry that has never been saved.

    Derived from content rather than random so two reads of an unchanged file
    agree — the UI edits one row at a time and a list position is not a handle
    (delete row 2 and every later row's identity shifts). The moment a row is
    saved the id is written to the file and stops depending on content.
    """
    material = f"{index}|{entry.get('question', '')}|{entry.get('relevant_citations', [])}"
    return hashlib.blake2s(material.encode("utf-8"), digest_size=6).hexdigest()


def _problem(check: str, message: str, severity: str = "warning") -> Problem:
    return Problem(check=check, severity=severity, message=message)


def _normalise(entry: dict, line: int) -> list[Problem]:
    """Repair an entry's *shape* in place. Returns what had to be repaired.

    The split that matters: wrong **types** are coerced, because the UI cannot
    render a string where it expects a list and the author gains nothing from
    a stack trace. Wrong **values** — an unknown ``kind`` or ``status`` — are
    left exactly as written and merely reported, because silently rewriting
    someone's ``"lookup"`` to ``"prose"`` would hide a decision they got wrong
    rather than surface it.

    ``status`` defaults to ``"draft"`` where :func:`eval.goldset.load` defaults
    it to ``"approved"``. The difference is deliberate and one-way: this is
    the queue a button copies into the measured set, so an entry nobody has
    explicitly approved must not arrive pre-approved. An *unrecognised* status
    is likewise never "approved", so it cannot be promoted either.
    """
    problems: list[Problem] = []

    if not isinstance(entry.get("question"), str):
        if entry.get("question") is not None:
            problems.append(
                _problem("draft:question-type", f"line {line}: question was not a string")
            )
        entry["question"] = "" if entry.get("question") is None else str(entry["question"])

    cites = entry.get("relevant_citations")
    if cites is None:
        entry["relevant_citations"] = []
    elif isinstance(cites, str):
        # A bare string iterates per character everywhere downstream, which
        # turns "4 CFR § 1.1" into eleven citations.
        problems.append(
            _problem(
                "draft:citations-type",
                f"line {line}: relevant_citations was the string {cites!r}; read as one citation",
            )
        )
        entry["relevant_citations"] = [cites]
    elif isinstance(cites, list):
        entry["relevant_citations"] = [str(c) for c in cites]
    else:
        problems.append(
            _problem(
                "draft:citations-type",
                f"line {line}: relevant_citations was a {type(cites).__name__}; read as empty",
            )
        )
        entry["relevant_citations"] = []

    if "kind" not in entry or entry["kind"] is None:
        entry["kind"] = "prose"
    elif entry["kind"] not in goldset.KINDS:
        problems.append(
            _problem(
                "draft:kind",
                f"line {line}: kind {entry['kind']!r} is not one of {goldset.KINDS} "
                f"— left as written; fix it before approving",
            )
        )

    if "status" not in entry or entry["status"] is None:
        entry["status"] = "draft"
    elif entry["status"] not in goldset.STATUSES:
        problems.append(
            _problem(
                "draft:status",
                f"line {line}: status {entry['status']!r} is not one of {goldset.STATUSES} "
                f"— treated as not approved",
            )
        )

    if not isinstance(entry.get("notes"), str):
        entry["notes"] = "" if entry.get("notes") is None else str(entry["notes"])

    return problems


def find_draft(entries: list[dict], draft_id: str) -> dict | None:
    return next((e for e in entries if e.get("draft_id") == draft_id), None)


def scaffold(
    spec: TitleSpec,
    *,
    eval_dir: Path | str = DEFAULT_EVAL_DIR,
    db_path: Path | str = DEFAULT_DB,
    n_prose: int = goldset.MIN_PROSE_PER_TITLE,
    n_citation: int = goldset.MIN_CITATION_PER_TITLE,
    n_negatives: int = goldset.MIN_NEGATIVES_PER_TITLE,
    corpus: list[str] | None = None,
) -> list[dict]:
    """Create the draft queue for a title from ``eval.goldset.draft_for_title``.

    Refuses to overwrite an existing draft file: it may hold hours of
    unpromoted authoring, and regenerating is not worth the risk of losing it.
    """
    path = draft_path(spec.number, eval_dir)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists with {len(goldset.load(path))} entries; "
            f"promote or delete it before scaffolding again"
        )
    entries = goldset.draft_for_title(
        spec,
        db_path=Path(db_path),
        n_prose=n_prose,
        n_citation=n_citation,
        n_negatives=n_negatives,
        corpus=corpus,
    )
    for entry in entries:
        entry["draft_id"] = uuid.uuid4().hex[:12]
        entry["notes"] = strip_body_excerpt(str(entry.get("notes", "")))
    save_draft(spec.number, entries, eval_dir)
    return entries


def strip_body_excerpt(notes: str) -> str:
    """Remove the body excerpt ``draft_prose_stubs`` pastes into its notes.

    The CLI scaffolder inlines the first 600 characters of the section because
    a terminal author has no other way to see what they are writing about. In
    the UI that would defeat the whole collapsed-by-default arrangement: the
    text would be sitting in the notes box, already read, before the author
    has decided to open it. The CLI keeps its excerpt; the UI replaces it with
    a pointer to the Reveal Text button.
    """
    return _BODY_EXCERPT.sub(" Use “Reveal text” if the heading is not enough.", notes).strip()


_BODY_EXCERPT = re.compile(r"\s*Body:.*$", re.DOTALL)


def new_entry(citation: str = "", kind: str = "prose") -> dict:
    return {
        "question": "",
        "relevant_citations": [citation] if citation else [],
        "kind": kind,
        "notes": "",
        "status": "draft",
        "draft_id": uuid.uuid4().hex[:12],
    }


class StatusRefused(ValueError):
    """A bulk status change that was refused as a whole. ``reasons`` per row."""

    def __init__(self, reasons: list[tuple[str, str]]):
        self.reasons = reasons
        super().__init__(
            "nothing changed: " + "; ".join(f"{rid}: {why}" for rid, why in reasons)
        )


def set_status(
    title: int,
    draft_ids: Iterable[str],
    status: str,
    *,
    eval_dir: Path | str = DEFAULT_EVAL_DIR,
    db_path: Path | str = DEFAULT_DB,
) -> list[str]:
    """Set ``status`` on several draft rows in one write. Returns the ids changed.

    All-or-nothing, like promotion: if any selected row cannot take the status
    — an unknown id, an empty question, a citation the index does not have —
    :class:`StatusRefused` lists every reason and the file is not touched.
    Only ``status`` is edited. Like saving a single row, the write also
    persists the in-memory repairs and derived ids from reading the file.

    This is still only the draft. Nothing reaches gold.jsonl until Approve &
    merge validates it.
    """
    if status not in goldset.STATUSES:
        raise StatusRefused([("*", f"status must be one of {goldset.STATUSES}")])
    wanted = list(dict.fromkeys(str(i) for i in draft_ids))
    if not wanted:
        raise StatusRefused([("*", "no rows were selected")])

    with _DRAFT_LOCK:
        draft = read_draft(title, eval_dir)
        if draft.skipped:
            raise StatusRefused([(
                "*",
                f"{draft.skipped} line(s) of {draft.path.name} cannot be read; saving would "
                f"delete them — fix them by hand first",
            )])
        by_id = {e["draft_id"]: e for e in draft.entries}
        indexed = goldset.index_sections(Path(db_path)) if status == "approved" else set()

        reasons: list[tuple[str, str]] = []
        for rid in wanted:
            entry = by_id.get(rid)
            if entry is None:
                reasons.append((rid, "no such draft entry"))
            elif status == "approved":
                if not str(entry.get("question", "")).strip():
                    reasons.append((rid, "no question has been written yet"))
                if entry.get("kind") not in goldset.KINDS:
                    reasons.append((rid, f"kind {entry.get('kind')!r} is not one of "
                                         f"{goldset.KINDS}"))
                missing = [c for c in entry.get("relevant_citations") or [] if c not in indexed]
                if missing:
                    reasons.append((rid, f"{', '.join(missing)} is not in the index, so the "
                                         f"overlap meter could not check this question"))
        if reasons:
            raise StatusRefused(reasons)

        changed = [rid for rid in wanted if by_id[rid].get("status") != status]
        for rid in changed:
            by_id[rid]["status"] = status
        if changed:
            save_draft(title, draft.entries, eval_dir)
        return changed


# --- promotion --------------------------------------------------------------


_PROMOTE_LOCK = threading.Lock()


@contextlib.contextmanager
def gold_lock(gold_path: Path) -> Iterator[None]:
    """Serialise promotion: a thread lock, plus an advisory file lock.

    The file lock covers a second dashboard process. It does **not** cover
    ``python -m eval.goldset`` or a text editor, which take no lock — so the
    real protection is that promotion re-reads the gold file inside the lock
    and re-validates the merged result, keeping the read-modify-write window
    to microseconds rather than the length of an editing session.
    """
    with _PROMOTE_LOCK:
        lock_path = Path(gold_path).with_suffix(Path(gold_path).suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = None
        try:
            handle = open(lock_path, "w", encoding="utf-8")
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):  # pragma: no cover - non-posix or NFS
                logger.debug("advisory file lock unavailable; thread lock only")
            yield
        finally:
            if handle is not None:
                handle.close()


def for_gold(entry: dict) -> dict:
    """The entry as it should appear in gold.jsonl: no authoring bookkeeping."""
    clean = {k: v for k, v in entry.items() if k not in DRAFT_ONLY_FIELDS}
    clean["question"] = str(clean.get("question", "")).strip()
    clean["status"] = "approved"
    clean.setdefault("kind", "prose")
    clean.setdefault("relevant_citations", [])
    return clean


def _gold_form(row: dict) -> dict:
    """A gold row in the shape :func:`for_gold` produces, for comparison.

    Keeps the row's real status: a gold entry parked as ``draft`` is not the
    same thing as an approved draft entry with the same question.
    """
    clean = for_gold(row)
    clean["status"] = row.get("status", "approved")
    return clean


def _differing_fields(candidate: dict, gold_row: dict) -> tuple[str, ...]:
    """Fields that differ. An absent field and an empty one are the same:
    reading a draft fills in ``notes: ""``, which is not a change of content."""
    ours, theirs = candidate, _gold_form(gold_row)

    def value(d: dict, k: str):
        v = d.get(k)
        return None if v in ("", None) else v

    keys = ours.keys() | theirs.keys()
    return tuple(sorted(k for k in keys if value(ours, k) != value(theirs, k)))


@dataclass(frozen=True)
class Promotion:
    """What ``promote`` did, or refused to do and why."""

    ok: bool
    promoted: tuple[dict, ...] = ()
    already_present: tuple[dict, ...] = ()
    still_draft: int = 0
    errors: tuple[Problem, ...] = ()
    warnings: tuple[Problem, ...] = ()
    pre_existing: tuple[Problem, ...] = ()
    bad_citations: tuple[tuple[str, str], ...] = ()  # (question, citation)
    # (question, fields that differ from the gold row with the same question)
    conflicts: tuple[tuple[str, tuple[str, ...]], ...] = ()
    gold_before: int = 0
    gold_after: int = 0
    staleness: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        if self.ok and not self.promoted:
            return (
                f"nothing new to merge; retired {len(self.already_present)} draft entry(s) "
                f"already in the gold set, identical field for field"
            )
        if self.ok:
            return (
                f"promoted {len(self.promoted)} question(s); "
                f"gold set {self.gold_before} → {self.gold_after}"
            )
        return f"nothing written: {len(self.errors)} error(s)"


def _refuse(check: str, message: str, **kw) -> Promotion:
    return Promotion(ok=False, errors=(Problem(check, "error", message),), **kw)


def promote(
    title: int,
    *,
    specs: list[TitleSpec],
    eval_dir: Path | str = DEFAULT_EVAL_DIR,
    gold_path: Path | str = goldset.DEFAULT_GOLD,
    db_path: Path | str = DEFAULT_DB,
) -> Promotion:
    """Validate the approved drafts for one title and merge them into gold.jsonl.

    All-or-nothing, across **both** files: either gold.jsonl gains the
    approved entries *and* the draft loses them, or neither file changes.

    The draft is the only copy of unmerged work, so it is the file that must
    never be rewritten on a guess. Three rules follow:

    * A draft row is retired only when gold.jsonl, as verified on disk, holds
      it field for field — appended by this call, or already there identical
      (the re-run after a crash between the two writes). A row whose question
      is already in gold but whose citations, kind or notes differ is a
      *conflict*: promotion refuses rather than silently keeping one version.
    * Gold is written first, then read back and compared. Only then is the
      draft rewritten — and only if its bytes are still exactly what was read.
    * Any failure after gold was written restores gold.jsonl to its original
      bytes, so a failed promotion leaves both files as they were.

    Pre-existing problems in gold.jsonl are reported but do not block. A stale
    negative someone else left behind must not make the file unappendable
    forever — the gate's job is to stop *new* breakage getting in.
    """
    gold_path = Path(gold_path)
    dpath = draft_path(title, eval_dir)
    with gold_lock(gold_path), _DRAFT_LOCK:
        sections = goldset.index_sections(Path(db_path))
        if not sections:
            return _refuse(
                "authoring:no-index",
                f"no indexed sections in {db_path} — a citation cannot be verified "
                f"against an index that is not built, and an unverified citation is "
                f"a question that can never be answered correctly",
            )

        # Parse exactly the bytes we will later check, so "unchanged since we
        # read it" means what it says.
        try:
            draft_original = _snapshot(dpath)
        except OSError as exc:
            return _refuse("authoring:draft-unreadable", f"cannot read {dpath}: {exc}")
        draft_file = _parse_draft(dpath, (draft_original or b"").decode("utf-8", "replace"))
        if draft_file.skipped:
            # Rewriting the draft would drop these lines for good.
            unreadable = [str(p) for p in draft_file.problems
                          if p.check in ("draft:bad-json", "draft:not-an-object")]
            return _refuse(
                "authoring:draft-unreadable-lines",
                f"{draft_file.skipped} line(s) of {dpath.name} could not be read "
                f"({'; '.join(unreadable)}). Promotion rewrites the draft, which would "
                f"delete them — fix them by hand first",
            )
        draft = draft_file.entries

        approved = [e for e in draft if e.get("status") == "approved"]
        still_draft = len(draft) - len(approved)
        if not approved:
            return _refuse(
                "authoring:nothing-approved",
                f"no entry in the Title {title} draft is marked approved "
                f"({still_draft} still in draft)",
                still_draft=still_draft,
            )

        try:
            gold_original = _snapshot(gold_path)
            gold = goldset.load(gold_path)
        except (OSError, ValueError) as exc:
            return _refuse("authoring:gold-unreadable", f"cannot read {gold_path}: {exc}",
                           still_draft=still_draft)
        if _snapshot(gold_path) != gold_original:  # pragma: no cover - lockless writer
            return _refuse("authoring:gold-changed",
                           f"{gold_path} changed while it was being read — try again",
                           still_draft=still_draft)

        by_question: dict[str, dict] = {}
        for row in gold:
            by_question.setdefault(str(row.get("question", "")).strip(), row)

        candidates = [for_gold(e) for e in approved]
        already, conflicts, incoming = [], [], []
        for c in candidates:
            row = by_question.get(c["question"])
            if row is None:
                incoming.append(c)
            elif differing := _differing_fields(c, row):
                conflicts.append((c["question"], differing))
            else:
                already.append(c)

        if conflicts:
            return Promotion(
                ok=False,
                errors=(Problem(
                    "authoring:conflicts-with-gold", "error",
                    f"{len(conflicts)} approved draft entry(s) ask a question that is already "
                    f"in {gold_path.name} but differ from it. Retiring them would discard "
                    f"the draft's version; appending them would duplicate the question. "
                    f"Make the draft match gold, or delete those draft rows",
                ),),
                conflicts=tuple(conflicts),
                still_draft=still_draft,
                gold_before=len(gold),
                gold_after=len(gold),
            )

        # Requirement 2b, stated per entry so the UI can mark the offending
        # row rather than printing a wall of text.
        bad = tuple(
            (c["question"], cite)
            for c in incoming
            for cite in c.get("relevant_citations") or []
            if cite not in sections
        )

        baseline = {str(p) for p in goldset.validate(gold, sections=sections, specs=specs)}
        merged = gold + incoming
        problems = goldset.validate(merged, sections=sections, specs=specs)
        new_problems = [p for p in problems if str(p) not in baseline]
        errors = tuple(p for p in new_problems if p.severity == "error")
        warnings = tuple(p for p in new_problems if p.severity != "error")
        pre_existing = tuple(p for p in problems if str(p) in baseline)

        if errors or bad:
            return Promotion(
                ok=False,
                errors=errors,
                warnings=warnings,
                pre_existing=pre_existing,
                bad_citations=bad,
                still_draft=still_draft,
                gold_before=len(gold),
                gold_after=len(gold),
            )

        promoted_ids = {e["draft_id"] for e in approved}
        remaining = [e for e in draft if e["draft_id"] not in promoted_ids]
        failure = _commit(
            gold_path, gold_original, merged if incoming else None,
            dpath, draft_original, remaining,
        )
        if failure is not None:
            return Promotion(
                ok=False,
                errors=(failure,),
                warnings=warnings,
                pre_existing=pre_existing,
                still_draft=still_draft,
                gold_before=len(gold),
                gold_after=len(gold),
            )

        return Promotion(
            ok=True,
            promoted=tuple(incoming),
            already_present=tuple(already),
            still_draft=still_draft,
            warnings=warnings,
            pre_existing=pre_existing,
            gold_before=len(gold),
            gold_after=len(merged),
            staleness=tuple(_staleness_after(specs, merged)),
        )


class _DraftChanged(Exception):
    pass


def _commit(
    gold_path: Path,
    gold_original: bytes | None,
    merged: list[dict] | None,
    dpath: Path,
    draft_original: bytes | None,
    remaining: list[dict],
) -> Problem | None:
    """Write gold (if ``merged``), verify it, then the draft; undo gold on any failure.

    Returns ``None`` on success, else the error to report. On every error path
    the draft is untouched, and gold is restored to ``gold_original`` unless the
    restore itself fails — which is reported, and is still safe: the draft keeps
    its approved rows, and re-running finds them in gold identical and retires
    them instead of appending them twice.
    """

    def rolled_back(what: str, exc: BaseException) -> Problem:
        try:
            _restore(gold_path, gold_original)
            undo = f"{gold_path.name} has been restored to exactly what it was"
        except OSError as restore_exc:
            logger.error("promotion rollback failed: %s", restore_exc)
            undo = (
                f"restoring {gold_path.name} ALSO failed ({restore_exc}), so it now holds "
                f"the merged entries while the draft still holds them as approved. Nothing "
                f"is lost: run Approve & merge again and it will retire the duplicates"
            )
        return Problem(
            "authoring:write-failed", "error",
            f"{what}: {exc}. The draft was not modified, and {undo}",
        )

    if merged is not None:
        try:
            _atomic_write(gold_path, _as_jsonl(merged))
            if goldset.load(gold_path) != merged:
                raise OSError(f"{gold_path.name} did not read back as written")
        except (OSError, ValueError) as exc:
            return rolled_back(f"writing {gold_path.name} failed", exc)

    try:
        if _snapshot(dpath) != draft_original:
            raise _DraftChanged(f"{dpath.name} was edited while the merge was running")
        _atomic_write(dpath, _as_jsonl(remaining))
    except (OSError, _DraftChanged) as exc:
        return rolled_back(f"updating {dpath.name} failed", exc)
    return None


def _staleness_after(specs: list[TitleSpec], merged: list[dict]) -> list[str]:
    """Why params.json no longer describes the gold set we just wrote.

    Derived, never stored: the banner is recomputed from the files on every
    request, so it cannot drift out of sync with what is actually on disk.
    """
    from rag import reranker as reranker_mod
    from retrieval import params as params_mod
    from retrieval import tune as tune_mod

    _, prov = params_mod.load()
    approved = [e for e in merged if e.get("status", "approved") == "approved"]
    try:
        return tune_mod.staleness(prov, specs, approved, reranker_mod.DEFAULT_RERANK_MODEL)
    except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - defensive
        logger.warning("could not compute staleness: %s", exc)
        return []


# --- queue view -------------------------------------------------------------


@dataclass(frozen=True)
class DraftItem:
    entry: dict
    section: Section
    overlap: Overlap

    @property
    def draft_id(self) -> str:
        return str(self.entry.get("draft_id", ""))

    @property
    def is_negative(self) -> bool:
        return not self.entry.get("relevant_citations")

    @property
    def ready(self) -> bool:
        return bool(str(self.entry.get("question", "")).strip())

    @property
    def unverified(self) -> bool:
        """Cites a section the index does not have.

        The dangerous row: the meter cannot say anything about it, and
        promotion will refuse it. Both facts should be visible while the
        author is writing, not discovered at merge time.
        """
        return bool(self.entry.get("relevant_citations")) and not self.section.found

    @property
    def approvable(self) -> bool:
        return not self.unverified


def meter_for(entry: dict, section: Section, vocab: Vocabulary) -> Overlap:
    """The meter state for one entry — always explicit, never a false green.

    Every branch that cannot produce a real score says so in its own words.
    The one thing this function will not do is return a zero-overlap GREEN
    for a question it never compared against anything.
    """
    question = str(entry.get("question", "")).strip()
    citations = entry.get("relevant_citations") or []

    if not citations:
        return not_applicable(
            "Negative question: no section is expected to answer it, so there is no "
            "vocabulary to borrow and nothing for the meter to check."
        )
    if not section.found:
        return overlap(
            question or "placeholder", "",
            unchecked_reason=(
                f"{citations[0]} is not in the index, so this question has NOT been "
                f"checked for borrowed wording. Index the title first — and note that "
                f"promotion will refuse this entry until the citation resolves."
            ),
        )
    if not question:
        return overlap("", section.body, vocab=vocab, kind=entry.get("kind", "prose"))
    if not section.body.strip():
        return overlap(
            question, "",
            unchecked_reason=(
                f"{section.citation} is in the index but its stored text is empty, so "
                f"this question has NOT been checked. That is an indexing bug worth "
                f"reporting, not a clean result."
            ),
        )
    return overlap(question, section.body, vocab=vocab, kind=entry.get("kind", "prose"))


def queue(
    title: int,
    *,
    eval_dir: Path | str = DEFAULT_EVAL_DIR,
    db_path: Path | str = DEFAULT_DB,
) -> list[DraftItem]:
    """The draft queue for a title, each entry joined to its section and meter."""
    entries = load_draft(title, eval_dir)
    citations = [
        (e.get("relevant_citations") or [None])[0] for e in entries if e.get("relevant_citations")
    ]
    sections = load_sections(db_path, [c for c in citations if c])
    vocab = vocabulary(db_path)
    items = []
    for entry in entries:
        cites = entry.get("relevant_citations") or []
        section = sections.get(cites[0], MISSING_SECTION) if cites else MISSING_SECTION
        items.append(
            DraftItem(entry=entry, section=section, overlap=meter_for(entry, section, vocab))
        )
    return items


# --- calibration ------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """What the approved gold set says the warning bands should be."""

    scores: tuple[tuple[float, str], ...]  # (distinctive, question), worst first
    rare_counts: tuple[int, ...]  # terms of art copied, per question
    n_sections: int
    n_scored: int
    n_skipped: int
    corpus: tuple[str, ...]
    gold_sha256: str

    @property
    def values(self) -> list[float]:
        return [v for v, _ in self.scores]

    def percentile(self, pct: float) -> float:
        return _percentile(self.values, pct)

    def flagged(self, amber: float, red: float, rare_red: int) -> tuple[int, int]:
        """How many already-approved questions these bands would call amber / red.

        The sanity check that matters. These questions passed human review, so
        bands that paint most of them red are measuring the wrong thing — or
        the band is simply too tight to be useful as a warning.
        """
        reds = sum(
            1 for (v, _), rc in zip(self.scores, self.rare_counts, strict=True)
            if v >= red or rc >= rare_red
        )
        ambers = sum(
            1 for (v, _), rc in zip(self.scores, self.rare_counts, strict=True)
            if not (v >= red or rc >= rare_red) and (v >= amber or rc)
        )
        return ambers, reds

    def propose(self) -> dict[str, float]:
        """Bands read off the distribution rather than guessed.

        Amber at the 75th percentile and red at the 95th: a warning that fires
        on a quarter of *approved* work is a warning people learn to ignore,
        and one that fires on nothing never fires at all. The top 5% of the
        approved set really are its most derivative questions — the listing
        below shows them, and they are worth rewriting.
        """
        return {
            "DISTINCTIVE_AMBER": round(self.percentile(75), 2),
            "DISTINCTIVE_RED": round(self.percentile(95), 2),
        }


def calibrate(
    gold_path: Path | str = goldset.DEFAULT_GOLD,
    db_path: Path | str = DEFAULT_DB,
    manifest_path: Path | str = manifest.DEFAULT_MANIFEST,
) -> Calibration:
    """Score every approved prose question against the section it cites.

    This is how the thresholds should be set. The approved set is the best
    available picture of what "good" looks like here: its authors were
    already trying to de-vocab, and their notes say so. A band that most of
    them fail is a band that is wrong about this corpus.

    Only prose counts. A citation lookup is *supposed* to quote the section
    number, so including them would drag the distribution toward the one kind
    of question the meter deliberately does not judge.
    """
    vocab = vocabulary(db_path)
    gold = [
        e
        for e in goldset.load(gold_path)
        if e.get("status", "approved") == "approved"
        and e.get("kind", "prose") == "prose"
        and e.get("relevant_citations")
    ]
    sections = load_sections(db_path, [e["relevant_citations"][0] for e in gold])
    scored: list[tuple[float, str]] = []
    rare_counts: list[int] = []
    skipped = 0
    for entry in gold:
        section = sections.get(entry["relevant_citations"][0])
        if section is None or not section.found or not section.body.strip():
            skipped += 1  # unindexed: cannot be scored, must not be counted as 0.0
            continue
        score = overlap(entry["question"], section.body, vocab=vocab)
        if not score.checked:
            skipped += 1
            continue
        scored.append((score.distinctive, entry["question"]))
        rare_counts.append(len(score.rare_copied))

    order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
    try:
        specs = manifest.load(manifest_path)
        corpus = tuple(manifest.corpus_state(specs))
    except (OSError, manifest.ManifestError):
        corpus = ()
    return Calibration(
        scores=tuple(scored[i] for i in order),
        rare_counts=tuple(rare_counts[i] for i in order),
        n_sections=vocab.n_sections,
        n_scored=len(scored),
        n_skipped=skipped,
        corpus=corpus,
        gold_sha256=eval_run.gold_fingerprint(gold)["sha256"] if gold else "",
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def rare_term_report(db_path: Path | str = DEFAULT_DB) -> list[tuple[int, float, int]]:
    """``(df, rarity, how many corpus terms are at least that rare)``.

    Grounds :data:`RARE_TERM` in the corpus rather than in a round number: it
    answers "a term this rare appears in at most N sections, and there are M
    such terms", which is what "term of art" has to mean here.
    """
    vocab = vocabulary(db_path)
    if vocab.n_sections <= 1:
        return []
    out = []
    for df in (1, 2, 3, 5, 10, 20, 50):
        rarity = vocab.rarity_for_df(df)
        count = sum(1 for v in vocab.df.values() if v <= df)
        out.append((df, rarity, count))
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(prog="python -m api.authoring", description=__doc__)
    parser.add_argument("--calibrate", action="store_true", help="Score the approved gold set")
    parser.add_argument("--gold", type=Path, default=goldset.DEFAULT_GOLD)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=manifest.DEFAULT_MANIFEST)
    parser.add_argument("--worst", type=int, default=12, help="How many questions to list")
    args = parser.parse_args(argv)

    if not args.calibrate:
        parser.print_help()
        return 0

    cal = calibrate(args.gold, args.db, args.manifest)
    if not cal.n_scored:
        print("no approved prose question could be scored — is the index built?")
        return 2

    print(f"corpus:  {', '.join(cal.corpus) or '(unknown)'}")
    print(f"gold:    {cal.gold_sha256} — {cal.n_scored} approved prose questions scored"
          + (f", {cal.n_skipped} unscorable (section not indexed)" if cal.n_skipped else ""))
    print(f"vocab:   {cal.n_sections} sections\n")

    print("distinctive-overlap distribution over ALREADY-APPROVED questions")
    for pct in (10, 25, 50, 75, 90, 95, 99, 100):
        print(f"  p{pct:<4} {cal.percentile(pct):.3f}")

    print(f"\nterms of art copied per question (threshold rarity >= {RARE_TERM})")
    counts: Counter[int] = Counter(cal.rare_counts)
    for n in sorted(counts):
        share = 100 * counts[n] / cal.n_scored
        print(f"  {n} term(s): {counts[n]:>4} questions ({share:.0f}%)")

    print("\nwhat 'rare' means on this corpus")
    print(f"  {'df':>4}  {'rarity':>7}  {'terms at or below':>18}")
    for df, rarity, count in rare_term_report(args.db):
        print(f"  {df:>4}  {rarity:>7.3f}  {count:>18}")

    ambers, reds = cal.flagged(DISTINCTIVE_AMBER, DISTINCTIVE_RED, RARE_COPIED_RED)
    print(f"\ncurrent bands  amber={DISTINCTIVE_AMBER} red={DISTINCTIVE_RED} "
          f"rare_copied_red={RARE_COPIED_RED}")
    print(f"  would flag {ambers} amber ({100 * ambers / cal.n_scored:.0f}%) and "
          f"{reds} red ({100 * reds / cal.n_scored:.0f}%) of the approved set")

    proposed = cal.propose()
    p_amber, p_red = proposed["DISTINCTIVE_AMBER"], proposed["DISTINCTIVE_RED"]
    p_ambers, p_reds = cal.flagged(p_amber, p_red, RARE_COPIED_RED)
    print(f"\nproposed bands amber={p_amber} (p75) red={p_red} (p95)")
    print(f"  would flag {p_ambers} amber ({100 * p_ambers / cal.n_scored:.0f}%) and "
          f"{p_reds} red ({100 * p_reds / cal.n_scored:.0f}%) of the approved set")

    print(f"\nmost derivative approved questions (top {args.worst}) — worth rewriting:")
    for (score, question), rare in list(zip(cal.scores, cal.rare_counts, strict=True))[
        : args.worst
    ]:
        print(f"  {score:.3f}  ({rare} rare)  {question[:84]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
