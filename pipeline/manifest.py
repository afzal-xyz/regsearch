"""The declared corpus: which CFR titles, at which issue dates, make up the index.

``corpus.toml`` at the repo root lists every title RegSearch is built from.
Every pipeline stage reads it and iterates over *all* entries rather than
whichever title the user happened to name, which is what stops a corpus from
drifting into a half-onboarded state — Title 1 indexed but absent from the
gold set, or Title 4 chunked at one date and indexed at another.

Reading uses stdlib ``tomllib``. Writing is a plain append of a fixed
three-line block: a TOML *writer* is a dependency, and this file's shape is
small enough that generating it by hand is safer than taking one on. The
cost is that comments and formatting elsewhere in the file are preserved
exactly (we never rewrite it), which is the behaviour we want anyway.

CLI:
    python -m pipeline.manifest            # show the declared corpus and its state
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from index.store import DEFAULT_DB
from ingest import fetcher
from ingest.chunker import DATA_CHUNKS

DEFAULT_MANIFEST = Path("corpus.toml")


class ManifestError(RuntimeError):
    """The manifest is missing, malformed, or self-contradictory."""


@dataclass(frozen=True, order=True)
class TitleSpec:
    number: int
    date: str
    name: str = ""

    @property
    def slug(self) -> str:
        """Stable identifier for one title at one date, e.g. ``4@2024-07-18``.

        Used wherever a corpus state has to be recorded inside a data file —
        notably the gold set's record of which corpus its negatives were
        verified against (see :mod:`eval.goldset`).
        """
        return f"{self.number}@{self.date}"

    @property
    def label(self) -> str:
        return f"Title {self.number}" + (f" ({self.name})" if self.name else "")

    def raw_path(self, data_dir: Path = fetcher.DATA_RAW) -> Path:
        return fetcher.cache_path(self.number, self.date, data_dir)

    def chunks_path(self, out_dir: Path = DATA_CHUNKS) -> Path:
        return Path(out_dir) / f"title-{self.number}-{self.date}.jsonl"


def load(path: Path | str = DEFAULT_MANIFEST) -> list[TitleSpec]:
    """Parse the manifest, sorted by title number. Raises on a malformed file."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"no manifest at {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{path} is not valid TOML: {exc}") from exc

    entries = data.get("titles")
    if not entries:
        raise ManifestError(f"{path} declares no [[titles]]")

    specs = []
    for i, entry in enumerate(entries):
        try:
            number = int(entry["number"])
            date = str(entry["date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"{path} [[titles]] #{i + 1} needs `number` and `date`") from exc
        fetcher._validate_date(date)
        specs.append(TitleSpec(number, date, str(entry.get("name", ""))))

    seen: dict[int, TitleSpec] = {}
    for spec in specs:
        if spec.number in seen:
            # Two dates for one title would index the same sections twice under
            # different vintages, and no stage downstream could tell which is
            # authoritative.
            raise ManifestError(
                f"{path} declares Title {spec.number} twice "
                f"({seen[spec.number].date} and {spec.date}); one date per title"
            )
        seen[spec.number] = spec
    return sorted(specs)


def corpus_state(specs: list[TitleSpec]) -> list[str]:
    """The corpus as a sorted list of ``title@date`` slugs."""
    return [s.slug for s in sorted(specs)]


def add(
    title: int,
    date: str,
    name: str = "",
    *,
    path: Path | str = DEFAULT_MANIFEST,
) -> tuple[list[TitleSpec], bool]:
    """Append a title to the manifest. Returns ``(specs, added)``.

    Idempotent: re-declaring a title at the same date is a no-op, so
    re-running an interrupted onboarding is safe. Declaring the *same* title
    at a *different* date is an error — changing a title's vintage means
    re-indexing it, which is a deliberate act, not an append.
    """
    path = Path(path)
    fetcher._validate_date(date)
    existing = load(path)
    for spec in existing:
        if spec.number == int(title):
            if spec.date == date:
                return existing, False
            raise ManifestError(
                f"Title {title} is already declared at {spec.date}, not {date}. "
                f"To change its vintage, edit {path} and re-run with --rebuild."
            )

    block = f'\n[[titles]]\nnumber = {int(title)}\ndate = "{date}"\n'
    if name:
        block += f'name = "{name}"\n'
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(block)
    return load(path), True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.manifest",
        description="Show the declared corpus and whether each stage's artifacts exist.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args(argv)

    try:
        specs = load(args.manifest)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    indexed: dict[int, int] = {}
    if args.db.exists():
        import sqlite3

        from index.store import connect

        try:
            conn = connect(args.db)
            try:
                indexed = {
                    int(t): n
                    for t, n in conn.execute("select title, count(*) from chunks group by title")
                }
            finally:
                conn.close()
        except sqlite3.Error:
            pass

    print(f"{'title':<8} {'date':<12} {'raw':>5} {'chunks':>7} {'indexed':>8}  name")
    for s in specs:
        print(
            f"{s.number:<8} {s.date:<12} "
            f"{'yes' if s.raw_path().exists() else '—':>5} "
            f"{'yes' if s.chunks_path().exists() else '—':>7} "
            f"{indexed.get(s.number, 0):>8}  {s.name}"
        )
    extra = sorted(set(indexed) - {s.number for s in specs})
    if extra:
        print(f"\nWARNING: indexed but not declared: {extra}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
