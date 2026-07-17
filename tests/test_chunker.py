"""Tests for ingest.chunker: whole-section default and long-section splitting."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest import chunker, parser
from ingest.parser import Section

FIXTURE = Path(__file__).parent / "fixtures" / "title-4-sample.xml"

CHUNK_KEYS = {
    "chunk_id",
    "citation",
    "section_citation",
    "heading",
    "heading_path",
    "title",
    "date",
    "paragraph_anchor",
    "char_start",
    "char_end",
    "token_estimate",
    "text",
}


@pytest.fixture
def sections() -> list[Section]:
    return parser.parse_file(FIXTURE, title=4, date="2024-07-18")


def test_estimate_tokens_monotonic():
    assert chunker.estimate_tokens("") == 0
    assert chunker.estimate_tokens("abcd") <= chunker.estimate_tokens("abcdefgh")


def test_short_section_single_chunk(sections):
    s21 = sections[0]
    chunks = chunker.chunk_section(s21, max_tokens=chunker.DEFAULT_MAX_TOKENS)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.citation == "4 CFR § 2.1"
    assert c.paragraph_anchor is None
    assert c.char_start == 0 and c.char_end == len(s21.body)
    # Text is pure body -- no citation or heading baked in.
    assert c.text == s21.body
    assert "CFR" not in c.text and "§" not in c.text


def test_long_section_splits_with_anchors(sections):
    s24 = sections[1]  # six paragraphs: (a)(b)(c)(1)(2)(3)
    chunks = chunker.chunk_section(s24, max_tokens=40)  # small budget -> forced split

    assert len(chunks) > 1
    # Every chunk keeps the base section citation as a prefix.
    assert all(c.section_citation == "4 CFR § 2.4" for c in chunks)
    assert all(c.citation.startswith("4 CFR § 2.4") for c in chunks)
    # First chunk starts at paragraph (a).
    assert chunks[0].paragraph_anchor == "(a)"
    assert chunks[0].citation == "4 CFR § 2.4(a)"
    # A later designator such as (b) shows up as an anchor.
    assert "4 CFR § 2.4(b)" in {c.citation for c in chunks}


def test_split_spans_are_ordered_in_bounds_and_pure(sections):
    s24 = sections[1]
    body = s24.body
    chunks = chunker.chunk_section(s24, max_tokens=40)

    prev_end = 0
    for c in chunks:
        assert 0 <= c.char_start < c.char_end <= len(body)
        assert c.char_start >= prev_end  # non-overlapping, increasing
        assert c.text == body[c.char_start : c.char_end]  # pure slice of body
        prev_end = c.char_end


def test_chunk_ids_unique_and_metadata_complete(sections):
    s24 = sections[1]
    chunks = chunker.chunk_section(s24, max_tokens=40)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "4-2024-07-18-2.4-000"
    for c in chunks:
        assert set(c.to_dict().keys()) == CHUNK_KEYS
        assert c.heading_path[-1] == c.heading  # invariant carried through


def test_empty_section_yields_no_chunks():
    s = Section(
        title=4,
        date="2024-07-18",
        number="1",
        citation="4 CFR § 1",
        heading="PART 1 [RESERVED]",
        heading_path=["Title 4—Accounts", "PART 1 [RESERVED]"],
        paragraphs=[],
    )
    assert chunker.chunk_section(s) == []


def test_chunk_title_writes_jsonl(tmp_path):
    import json

    # Stage the fixture where fetcher.cache_path expects the raw XML.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "title-4-2024-07-18.xml").write_bytes(FIXTURE.read_bytes())

    out_dir = tmp_path / "chunks"
    out_path, n = chunker.chunk_title(
        4, "2024-07-18", data_dir=raw_dir, out_dir=out_dir, max_tokens=40
    )

    assert out_path == out_dir / "title-4-2024-07-18.jsonl"
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n > 0
    first = json.loads(lines[0])
    assert set(first.keys()) == CHUNK_KEYS
    assert first["citation"].startswith("4 CFR § 2.1")
