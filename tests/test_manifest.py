"""Tests for the declared-corpus manifest (pipeline.manifest)."""

from __future__ import annotations

import pytest

from pipeline import manifest
from pipeline.manifest import ManifestError, TitleSpec

TWO_TITLES = """
[[titles]]
number = 4
date = "2024-07-18"
name = "Accounts"

[[titles]]
number = 31
date = "2024-07-18"
"""


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "corpus.toml"
    p.write_text(TWO_TITLES, encoding="utf-8")
    return p


def test_load_sorts_by_title_number(path):
    specs = manifest.load(path)
    assert [s.number for s in specs] == [4, 31]
    assert specs[0].name == "Accounts"
    assert specs[1].name == ""


def test_slug_and_corpus_state(path):
    specs = manifest.load(path)
    assert specs[0].slug == "4@2024-07-18"
    assert manifest.corpus_state(specs) == ["4@2024-07-18", "31@2024-07-18"]


def test_paths_follow_the_title_date_convention(tmp_path):
    spec = TitleSpec(4, "2024-07-18")
    assert spec.raw_path(tmp_path).name == "title-4-2024-07-18.xml"
    assert spec.chunks_path(tmp_path).name == "title-4-2024-07-18.jsonl"


def test_missing_manifest_is_an_error(tmp_path):
    with pytest.raises(ManifestError, match="no manifest"):
        manifest.load(tmp_path / "nope.toml")


def test_empty_manifest_is_an_error(tmp_path):
    p = tmp_path / "corpus.toml"
    p.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="declares no"):
        manifest.load(p)


def test_malformed_entry_is_an_error(tmp_path):
    p = tmp_path / "corpus.toml"
    p.write_text('[[titles]]\nnumber = 4\n', encoding="utf-8")
    with pytest.raises(ManifestError, match="needs `number` and `date`"):
        manifest.load(p)


def test_bad_date_is_an_error(tmp_path):
    p = tmp_path / "corpus.toml"
    p.write_text('[[titles]]\nnumber = 4\ndate = "July 2024"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        manifest.load(p)


def test_duplicate_title_is_an_error(tmp_path):
    """Two vintages of one title would double every section in the index."""
    p = tmp_path / "corpus.toml"
    p.write_text(
        '[[titles]]\nnumber = 4\ndate = "2024-07-18"\n'
        '[[titles]]\nnumber = 4\ndate = "2025-01-01"\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="twice"):
        manifest.load(p)


def test_add_appends_and_is_idempotent(path):
    specs, added = manifest.add(1, "2024-07-18", "General Provisions", path=path)
    assert added
    assert [s.number for s in specs] == [1, 4, 31]

    specs, added = manifest.add(1, "2024-07-18", path=path)
    assert not added, "re-declaring the same title/date must be a no-op"
    assert [s.number for s in specs] == [1, 4, 31]


def test_add_preserves_existing_comments(tmp_path):
    p = tmp_path / "corpus.toml"
    p.write_text("# keep me\n" + TWO_TITLES, encoding="utf-8")
    manifest.add(1, "2024-07-18", path=p)
    assert "# keep me" in p.read_text(encoding="utf-8")


def test_add_refuses_a_different_date_for_a_declared_title(path):
    with pytest.raises(ManifestError, match="already declared at"):
        manifest.add(4, "2025-01-01", path=path)
