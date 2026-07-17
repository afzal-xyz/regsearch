"""Tests for ingest.parser against a real Title 4 fragment fixture."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ingest import parser
from ingest.parser import Section

FIXTURE = Path(__file__).parent / "fixtures" / "title-4-sample.xml"


@pytest.fixture
def sections() -> list[Section]:
    return parser.parse_file(FIXTURE, title=4, date="2024-07-18")


def test_finds_both_sections(sections):
    assert [s.number for s in sections] == ["2.1", "2.4"]


def test_citation_format(sections):
    assert [s.citation for s in sections] == ["4 CFR § 2.1", "4 CFR § 2.4"]


def test_heading_path_title_down(sections):
    s21 = sections[0]
    assert s21.heading_path == [
        "Title 4—Accounts",
        "CHAPTER I—GOVERNMENT ACCOUNTABILITY OFFICE",
        "SUBCHAPTER A—PERSONNEL SYSTEM",
        "PART 2—PURPOSE AND GENERAL PROVISION",
        "§ 2.1 Purpose, scope, and applicability.",
    ]


def test_heading_path_invariant(sections):
    for s in sections:
        assert s.heading_path[-1] == s.heading


def test_body_paragraphs_preserved_and_labeled(sections):
    s21 = sections[0]
    assert len(s21.paragraphs) == 2
    assert s21.paragraphs[0].startswith("(a) This regulation establishes")
    assert s21.paragraphs[1].startswith("(b) Nothing in this regulation")


def test_metadata_excluded_from_body(sections):
    # § 2.4 has an AUTH/SOURCE at PART level and a trailing <CITA>; none belong
    # in the section body. Only the six lettered/numbered paragraphs remain.
    s24 = sections[1]
    assert len(s24.paragraphs) == 6
    body = s24.body
    assert "Authority:" not in body
    assert "45 FR 68375" not in body  # from SOURCE / CITA


def test_source_title_and_date_carried(sections):
    for s in sections:
        assert s.title == 4
        assert s.date == "2024-07-18"


def test_heading_path_invariant_enforced():
    with pytest.raises(ValueError):
        Section(
            title=4,
            date="2024-07-18",
            number="2.1",
            citation="4 CFR § 2.1",
            heading="§ 2.1 Purpose.",
            heading_path=["Title 4—Accounts"],  # tail != heading
        )


def test_unrecognized_block_element_warns(caplog):
    xml = (
        '<ECFR><DIV1 N="4" TYPE="TITLE"><HEAD>Title 4—Accounts</HEAD>'
        '<DIV8 N="9.9" TYPE="SECTION"><HEAD>§ 9.9 Test.</HEAD>'
        "<P>(a) A normal paragraph.</P>"
        "<GPOTABLE>some table markup</GPOTABLE>"
        "</DIV8></DIV1></ECFR>"
    )
    with caplog.at_level(logging.WARNING, logger="ingest.parser"):
        sections = parser.parse_title(xml, title=4, date="2024-07-18")

    # The unknown element is skipped from the body but logged, not silent.
    assert sections[0].paragraphs == ["(a) A normal paragraph."]
    assert any("GPOTABLE" in rec.message for rec in caplog.records)
