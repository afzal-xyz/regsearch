"""Chunk parsed eCFR sections into retrieval units and write JSONL.

Default: one chunk per section. A section whose body exceeds ``max_tokens`` is
split at paragraph boundaries into non-overlapping chunks, each still carrying
the section citation plus a paragraph anchor (e.g. ``"4 CFR § 7.3(b)"``).

Chunk ``text`` is PURE section body -- no citation or heading is baked in.
Contextualization (prepending citation/heading) happens at index time.

Anchor limitation: ``paragraph_anchor`` is the *leading* designator of a
chunk's first labeled paragraph. eCFR paragraphs nest -- e.g. ``(c)`` then
``(1)``, ``(2)`` beneath it. If a chunk begins at such a nested paragraph, the
anchor records the immediate marker (``(1)``) only, not the full nested path
(``(c)(1)``). Callers needing the exact nested address must recover it from the
surrounding section, not from the anchor alone.

CLI:
    python -m ingest.chunker --title 4 --date 2024-07-18
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ingest import fetcher
from ingest.parser import Section, parse_file

logger = logging.getLogger(__name__)

DATA_CHUNKS = Path("data/chunks")
DEFAULT_MAX_TOKENS = 1000

_PARA_SEP = "\n\n"
_LABEL_RE = re.compile(r"^\(([A-Za-z0-9]{1,4})\)")
_SLUG_RE = re.compile(r"[^A-Za-z0-9.]+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token).

    Good enough for chunk-boundary decisions and dependency-free. Swap in a real
    tokenizer later if exactness matters; the ``max_tokens`` target is approximate.
    """
    return math.ceil(len(text) / 4)


@dataclass
class Chunk:
    chunk_id: str
    citation: str  # section citation + anchor if split, e.g. "4 CFR § 2.4(b)"
    section_citation: str  # base section citation, e.g. "4 CFR § 2.4"
    heading: str
    heading_path: list[str]
    title: int
    date: str
    paragraph_anchor: str | None  # e.g. "(b)"; None for whole-section chunks
    char_start: int  # span into the section body
    char_end: int
    token_estimate: int
    text: str  # pure body text (no citation/heading)

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(number: str) -> str:
    return _SLUG_RE.sub("_", number).strip("_") or "x"


def _leading_anchor(paragraph: str) -> str | None:
    m = _LABEL_RE.match(paragraph)
    return f"({m.group(1)})" if m else None


def _paragraph_offsets(paragraphs: list[str]) -> list[tuple[int, int]]:
    """Char (start, end) of each paragraph within ``_PARA_SEP.join(paragraphs)``."""
    offsets: list[tuple[int, int]] = []
    pos = 0
    for i, para in enumerate(paragraphs):
        start = pos
        end = start + len(para)
        offsets.append((start, end))
        pos = end + (len(_PARA_SEP) if i < len(paragraphs) - 1 else 0)
    return offsets


def _make_chunk(
    section: Section, k: int, start: int, end: int, text: str, anchor: str | None
) -> Chunk:
    citation = f"{section.citation}{anchor}" if anchor else section.citation
    return Chunk(
        chunk_id=f"{section.title}-{section.date}-{_slug(section.number)}-{k:03d}",
        citation=citation,
        section_citation=section.citation,
        heading=section.heading,
        heading_path=section.heading_path,
        title=section.title,
        date=section.date,
        paragraph_anchor=anchor,
        char_start=start,
        char_end=end,
        token_estimate=estimate_tokens(text),
        text=text,
    )


def chunk_section(section: Section, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    """Chunk a single section: whole-section by default, else split by paragraph."""
    paragraphs = section.paragraphs
    if not paragraphs:
        return []  # reserved / empty section

    body = section.body
    if len(paragraphs) == 1 or estimate_tokens(body) <= max_tokens:
        return [_make_chunk(section, 0, 0, len(body), body, None)]

    offsets = _paragraph_offsets(paragraphs)
    chunks: list[Chunk] = []
    group: list[int] = []  # paragraph indices in the current chunk

    def flush() -> None:
        if not group:
            return
        start = offsets[group[0]][0]
        end = offsets[group[-1]][1]
        anchor = next(
            (a for i in group if (a := _leading_anchor(paragraphs[i]))), None
        )
        chunks.append(_make_chunk(section, len(chunks), start, end, body[start:end], anchor))

    for i in range(len(paragraphs)):
        if group:
            trial = body[offsets[group[0]][0] : offsets[i][1]]
            if estimate_tokens(trial) > max_tokens:
                flush()
                group = [i]
            else:
                group.append(i)
        else:
            group = [i]
    flush()
    return chunks


def chunk_title(
    title: int,
    date: str,
    *,
    data_dir: Path = fetcher.DATA_RAW,
    out_dir: Path = DATA_CHUNKS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    force: bool = False,
) -> tuple[Path, int]:
    """Parse the cached title XML (fetching if absent) and write chunk JSONL.

    Returns ``(output_path, chunk_count)``.
    """
    xml_path = fetcher.cache_path(title, date, data_dir)
    if force or not xml_path.exists():
        xml_path = fetcher.fetch_title_xml(title, date, data_dir=data_dir, force=force)

    sections = parse_file(xml_path, title, date)
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(chunk_section(section, max_tokens))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"title-{int(title)}-{date}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    logger.info("wrote %d chunks from %d sections -> %s", len(chunks), len(sections), out_path)
    return out_path, len(chunks)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fetcher._use_os_trust_store()  # honor OS trust store if a fetch is needed

    parser = argparse.ArgumentParser(
        prog="python -m ingest.chunker",
        description="Chunk cached eCFR title XML into JSONL retrieval units.",
    )
    parser.add_argument("--title", type=int, required=True, help="CFR title number, e.g. 4")
    parser.add_argument("--date", required=True, help="Issue date, YYYY-MM-DD")
    parser.add_argument("--data-dir", type=Path, default=fetcher.DATA_RAW)
    parser.add_argument("--out-dir", type=Path, default=DATA_CHUNKS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--force", action="store_true", help="Re-fetch the raw XML")
    args = parser.parse_args(argv)

    try:
        out_path, n = chunk_title(
            args.title,
            args.date,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            max_tokens=args.max_tokens,
            force=args.force,
        )
    except (ValueError, OSError, fetcher.FetchError, fetcher.requests.RequestException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{out_path}  ({n} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
