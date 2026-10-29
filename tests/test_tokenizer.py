"""Tests for the citation-aware BM25 analyzer (index.tokenizer)."""

from __future__ import annotations

import pytest

from index.tokenizer import RANGE_EXPANSION_CAP, bm25_document, stem, tokenize


class TestCitationsStayWhole:
    """The naive [a-z0-9]+ tokenizer shredded these into useless digits."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("§ 2.3(b)(1)", ["§2.3(b)(1)", "§2.3"]),
            ("§2.3", ["§2.3"]),
            ("sec. 2.3", ["§2.3"]),
            ("section 2.3", ["§2.3"]),
            ("4 CFR 25.14", ["4cfr25.14", "§25.14"]),
            ("4 CFR part 81", ["4cfr81", "§81"]),
            ("29 U.S.C. 206(d)", ["29usc206(d)", "29usc206"]),
            ("Pub. L. 95-454", ["publ95-454"]),
        ],
    )
    def test_citation_forms(self, text, expected):
        assert tokenize(text) == expected

    def test_every_spelling_of_a_section_collapses_to_one_token(self):
        """What makes a citation searchable: query and document must agree."""
        spellings = ["§ 25.14", "§25.14", "sec. 25.14", "section 25.14", "4 CFR 25.14", "25.14"]
        assert all("§25.14" in tokenize(s) for s in spellings)

    def test_subdivision_also_emits_its_parent_section(self):
        """A query for the section must match text citing one of its paragraphs."""
        assert "§2.3" in tokenize("see § 2.3(b)(1) for details")

    def test_statutes_do_not_collide_with_cfr_sections(self):
        """"29 U.S.C. 206" is a different thing from "4 CFR § 206"."""
        assert not set(tokenize("29 U.S.C. 206")) & set(tokenize("§ 206"))

    def test_citations_are_never_stemmed(self):
        # A trailing "s" on a citation token would otherwise be stripped.
        assert tokenize("§ 28.18") == ["§28.18"]
        assert tokenize("4 CFR 83.17") == ["4cfr83.17", "§83.17"]

    def test_prose_around_a_citation_still_tokenizes(self):
        # "what" and "does" are stopwords; the content words and the citation stay.
        assert tokenize("What does § 7.1 say about unions?") == [
            "§7.1", "say", "about", "union",
        ]

    def test_stopwords_are_dropped_but_content_words_stay(self):
        """rank_bm25 gives negative-IDF terms an epsilon floor that grows with
        the corpus; at 6k+ chunks stopword-heavy sections outscored a unique
        citation token. Stopwords leave the vocabulary entirely instead."""
        assert tokenize("the terms used in this part") == ["term", "used", "part"]

    def test_a_stopword_is_never_a_citation(self):
        # Stopword removal must not touch citation or statute tokens.
        assert tokenize("as provided in 29 U.S.C. 206") == ["provided", "29usc206"]


class TestSectionRanges:
    """"§§ 28.18 through 28.21" governs 28.19 and 28.20 too."""

    def test_expands_every_section_in_the_range(self):
        assert tokenize("§§ 28.42 through 28.45") == ["§28.42", "§28.43", "§28.44", "§28.45"]

    def test_interior_section_is_findable(self):
        """The bug: a lookup for 28.19 missed text that plainly governs it."""
        tokens = tokenize("processed in accordance with §§ 28.18 through 28.88")
        assert "§28.19" in tokens
        assert "§28.50" in tokens

    @pytest.mark.parametrize(
        "text",
        ["§§ 28.42 through 28.45", "§§ 28.42-28.45", "sections 28.42 to 28.45", "§§ 28.42–28.45"],
    )
    def test_range_spellings(self, text):
        assert "§28.44" in tokenize(text)

    def test_does_not_expand_across_parts(self):
        """"28.18 to 31.5" is not a section range; expanding it invents sections."""
        assert tokenize("§§ 28.18 through 31.5") == ["§28.18", "§31.5"]

    def test_oversized_range_falls_back_to_endpoints(self):
        """A malformed range must not emit tens of thousands of tokens."""
        assert tokenize("§§ 1.1 through 1.99999") == ["§1.1", "§1.99999"]

    def test_expansion_is_capped(self):
        span = RANGE_EXPANSION_CAP + 5
        assert tokenize(f"§§ 2.1 through 2.{span}") == ["§2.1", f"§2.{span}"]
        # just inside the cap still expands
        inside = RANGE_EXPANSION_CAP - 1
        assert len(tokenize(f"§§ 2.1 through 2.{inside}")) == inside

    def test_reversed_and_degenerate_ranges(self):
        assert tokenize("§§ 28.45 through 28.42") == ["§28.45", "§28.42"]
        assert tokenize("§§ 28.42 through 28.42") == ["§28.42", "§28.42"]

    def test_does_not_swallow_a_statute_or_public_law(self):
        assert tokenize("Pub. L. 95-454") == ["publ95-454"]
        assert tokenize("29 U.S.C. 206(d)") == ["29usc206(d)", "29usc206"]

    def test_comma_lists_still_work(self):
        assert set(tokenize("§§ 5.1, 5.2")) >= {"§5.1", "§5.2"}


class TestStemming:
    def test_folds_plurals(self):
        assert stem("employees") == stem("employee") == "employee"
        assert stem("firearms") == stem("firearm") == "firearm"
        assert stem("copies") == "copy"
        assert stem("witnesses") == "witness"

    def test_does_not_overstem(self):
        """Words a regulation distinguishes must not collapse together.

        A Porter stemmer merges all of these; Part 25 is about the GAO
        *Building*, and an "applicant" is not an "applicable" rule.
        """
        assert stem("building") != stem("build")
        assert stem("applicant") != stem("applicable")
        assert stem("representative") != stem("represent")
        assert stem("filing") != stem("filed")

    def test_leaves_non_plural_s_alone(self):
        for word in ("its", "business", "analysis", "status", "address", "premises"):
            assert stem(word) == word

    def test_leaves_short_words_alone(self):
        assert stem("gao") == "gao"

    def test_question_plural_matches_regulation_singular(self):
        """The mismatch that motivated stemming at all."""
        question = set(tokenize("Do employees get copies of decisions?"))
        regulation = set(tokenize("Each employee shall receive a copy of the decision."))
        assert {"employee", "copy", "decision"} <= question & regulation


class TestBm25Document:
    CHUNK = {
        "citation": "4 CFR § 25.14",
        "heading_path": ["Title 4—Accounts", "§ 25.14 Weapons and explosives."],
        "text": "No person shall carry or possess firearms.",
    }

    def test_section_is_findable_by_its_own_citation(self):
        """A section's body never contains its own number — the fix for that."""
        assert "§25.14" not in tokenize(self.CHUNK["text"])
        assert "§25.14" in tokenize(bm25_document(self.CHUNK))

    def test_indexes_citation_heading_and_body(self):
        tokens = tokenize(bm25_document(self.CHUNK))
        assert "4cfr25.14" in tokens  # full citation
        assert "weapon" in tokens  # heading, stemmed
        assert "firearm" in tokens  # body, stemmed
