"""Citation-aware tokenizer for the BM25 index.

The naive ``[a-z0-9]+`` tokenizer shreds exactly the strings a regulation
search has to get right::

    "§ 2.3(b)(1)"  ->  ["2", "3", "b", "1"]
    "4 CFR 25.14"  ->  ["4", "cfr", "25", "14"]

Those fragments are useless: "2" and "3" match every section in the corpus, so
a query naming a specific section cannot find it. This module instead keeps
citations whole and *normalizes* them, so that all the ways of writing the same
reference collapse onto one token::

    "§ 25.14"  "sec. 25.14"  "section 25.14"  "4 CFR 25.14"  "25.14"  ->  "§25.14"

Normalization is what makes a citation searchable: documents and queries run
through this same function, so a question asking about "4 CFR 25.14" matches a
section whose text says "§ 25.14". A subdivided citation also emits its parent
section (``§ 2.3(b)(1)`` -> ``§2.3(b)(1)`` *and* ``§2.3``), so a query for the
section finds text that only cites one of its paragraphs.

Ordinary words are stemmed (see :func:`stem`); citation tokens never are —
stemming "§25.14" or "29usc206" would be nonsense.
"""

from __future__ import annotations

import logging
import re

from index.embedder import contextualize

logger = logging.getLogger(__name__)

# Bump whenever the analyzer — :func:`bm25_document` or :func:`tokenize` —
# changes its output. A persisted BM25 index built by an older analyzer cannot
# be queried by a newer one (the terms no longer line up), so the store
# rebuilds when this does not match.
#   1: naive [a-z0-9]+ over chunk text
#   2: citation-aware + plural stemming
#   3: index citation and heading path alongside the text
#   4: section-range expansion; subdivided citations emit their parent
#   5: drop English stopwords (see _STOPWORDS)
ANALYZER_VERSION = 5


def bm25_document(chunk) -> str:
    """The text BM25 indexes for a chunk: citation, heading path, then body.

    A section's body almost never contains its own number — the text of
    ``4 CFR § 25.14`` opens "No person while entering..." and never says
    "25.14" (only 7 of Title 4's 245 chunks cite themselves). So indexing the
    body alone makes a section *unfindable by its own citation*, no matter how
    well the tokenizer handles citations.

    This mirrors what the embedder already does — :func:`contextualize` is the
    vector side's input — so both retrievers see the same document, plus the
    full ``N CFR § X.Y`` citation that the heading alone does not spell out.
    """
    citation = chunk["citation"] if isinstance(chunk, dict) else chunk.citation
    return f"{citation}\n{contextualize(chunk)}"

# --- citation grammar -------------------------------------------------------

# A trailing run of subdivisions: "(b)", "(b)(1)", "(d)(2)(iii)".
_SUBDIVISIONS = r"(?:\([0-9A-Za-z]{1,4}\))*"
# A section number: 2.3, 25.14, 28.120, 3110, 2000e, 601a.
_SECTION_NUM = r"\d+[a-z]?(?:\.\d+)*"
# How a range is written: "§§ 28.18 through 28.21", "28.42-28.45".
_RANGE_SEP = r"(?:through|thru|to|[-–—])"

# Most sections a "§§ X through Y" reference expands to. The widest real range
# in Title 4 spans 73 numbers (§§ 28.18 through 28.90), so a smaller cap would
# truncate legitimate ranges and leave their middle sections unfindable — the
# bug this expansion exists to fix. The cap is a guard against *malformed*
# input: a garbled "§§ 1.1 through 1.99999" would otherwise emit 100,000
# tokens into one document. At 100 the worst case adds about half a median
# chunk (207 tokens), which the index absorbs without distorting BM25's
# document-length normalization.
RANGE_EXPANSION_CAP = 100

# Scanned left to right, first match wins, so the specific citation forms must
# precede the bare-number and word fallbacks.
_SCANNER = re.compile(
    r"""
      (?P<cfr_title>\d+)\s*c\.?\s*f\.?\s*r\.?\s*(?:part\s+|§+\s*)?
        (?P<cfr_num>""" + _SECTION_NUM + r""")(?P<cfr_sub>""" + _SUBDIVISIONS + r""")
    | (?P<usc_title>\d+)\s*u\.?\s*s\.?\s*c\.?\s*(?:§+\s*)?
        (?P<usc_num>""" + _SECTION_NUM + r""")(?P<usc_sub>""" + _SUBDIVISIONS + r""")
    | pub\.?\s*l\.?\s*(?:no\.?\s*)?(?P<publ>\d+[-–]\d+)
    | (?:§+|sec(?:tions?|s?\.)|sections?)\s*
        (?P<range_lo>""" + _SECTION_NUM + r""")\s*""" + _RANGE_SEP + r"""\s*
        (?P<range_hi>""" + _SECTION_NUM + r""")
    | (?:§+|sec(?:tions?|s?\.)|sections?)\s*
        (?P<sec_num>""" + _SECTION_NUM + r""")(?P<sec_sub>""" + _SUBDIVISIONS + r""")
    | (?P<bare_num>\d+\.\d+(?:\.\d+)*)(?P<bare_sub>""" + _SUBDIVISIONS + r""")
    | (?P<word>[a-z0-9]+)
    """,
    re.VERBOSE,
)


def _section_tokens(number: str, subdivisions: str) -> list[str]:
    """Canonical tokens for a CFR section reference, parent first."""
    section = f"§{number}"
    if not subdivisions:
        return [section]
    # Emit the full reference and its parent section, so a query for the
    # section still matches text that cites only one of its paragraphs.
    return [section + subdivisions.lower(), section]


def _range_tokens(low: str, high: str) -> list[str]:
    """Expand "§§ 28.18 through 28.21" to a token for every section in it.

    Without this a range names only its endpoints, so looking up § 28.19 — a
    section the range plainly covers — misses the text that governs it.

    Expansion applies only when the endpoints differ in their last component
    alone ("28.18" to "28.21", not "28.18" to "31.5"), which is what a CFR
    section range means. Anything else, or a span wider than
    :data:`RANGE_EXPANSION_CAP`, falls back to the two endpoints: a partial
    expansion would be worse than none, silently covering the start of a range
    and not the rest.
    """
    endpoints = [f"§{low}", f"§{high}"]
    low_parts, high_parts = low.split("."), high.split(".")
    prefix = low_parts[:-1]
    # Same part, and both are real section numbers ("28.18", not a bare "18").
    if not prefix or prefix != high_parts[:-1]:
        return endpoints
    try:
        start, end = int(low_parts[-1]), int(high_parts[-1])
    except ValueError:  # a lettered section like "601a" — not a numeric range
        return endpoints
    if not 0 < end - start < RANGE_EXPANSION_CAP:
        if end - start >= RANGE_EXPANSION_CAP:
            logger.debug("not expanding oversized section range §§ %s to %s", low, high)
        return endpoints
    stem_ = ".".join(prefix)
    return [f"§{stem_}.{n}" for n in range(start, end + 1)]


# --- stopwords ---------------------------------------------------------------

# Dropped from both documents and queries. rank_bm25 clamps a negative-IDF
# term (one in more than half the corpus) to ``epsilon * average_idf`` instead
# of zero, so ubiquitous function words retain real weight — and that floor
# *grows* with the corpus, because rarer vocabulary pushes the average IDF up.
# On the 245-chunk Title 4 index this was noise; at 6,262 chunks (Titles 4+31)
# a stopword-heavy section outscored a document holding a unique citation
# token: "Where are the terms used in § 1010.615 defined?" ranked the one
# section containing §1010.615 at 71, under sections matching only
# "where/the/term/used". Removing stopwords from the vocabulary entirely is
# the fix the rank_bm25 API allows (its epsilon is not exposed per-index).
#
# The list is function words only: articles, prepositions, pronouns,
# auxiliaries, and the modals ("shall", "may", "must") that appear in nearly
# every regulatory sentence and so discriminate nothing. Content words stay,
# however common — "employee", "agency", "record" are what prose queries are
# made of. Citation tokens are never stopwords.
_STOPWORDS = frozenset(
    """
    a an and any are as at be been both but by can could did do does for from
    had has have how i if in into is it its may me must my no nor not of on or
    our shall should so such than that the their them then there these they
    this those to under upon was we were what when where which who whom whose
    why will with within would you your
    """.split()
)

# --- stemming ---------------------------------------------------------------

# Words whose final "s"/"es" is not a plural, or where stripping it changes the
# meaning. Checked before the suffix rules.
_NO_STEM = frozenset(
    {
        "its", "has", "was", "does", "goes", "us", "this", "thus", "gas",
        "yes", "less", "unless", "plus", "bias", "basis", "versus", "campus",
        "witness", "address", "process", "access", "express", "redress",
        "premises", "proceeds", "damages", "wages", "costs", "series",
    }
)


def stem(word: str) -> str:
    """Strip plural endings only — a deliberately conservative stemmer.

    This is an S-stemmer: it folds "employees" onto "employee" and "firearms"
    onto "firearm", which is the mismatch that actually costs recall when a
    question is phrased in the plural and the regulation is written in the
    singular.

    It stops there on purpose. Stripping "-ing"/"-ed" as well, or running a
    full Porter stemmer, conflates words that a regulation distinguishes: the
    "GAO Building" would stem to "build", "filing" to "fil", "meeting" to
    "meet". Measured on ``eval/gold.jsonl``, the heavier stemmers scored no
    better and some scored worse — see "BM25 tokenization" in the README.
    """
    if len(word) <= 3 or word in _NO_STEM or not word.isalpha():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        # "witnesses" -> "witness", "boxes" -> "box", but "cases" -> "case".
        if word[:-2].endswith(("ss", "sh", "ch", "x", "z", "s")):
            return word[:-2]
        return word[:-1]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


# --- public API -------------------------------------------------------------


def _scan(text: str, stemmer) -> list[tuple[str, bool]]:
    """Yield ``(token, is_citation)`` pairs for ``text``."""
    tokens: list[tuple[str, bool]] = []

    def cite(*values: str) -> None:
        tokens.extend((v, True) for v in values)

    for m in _SCANNER.finditer(text.lower()):
        kind = m.lastgroup
        if m.group("word") is not None:
            word = m.group("word")
            if word in _STOPWORDS:
                continue
            tokens.append(((stemmer(word) if stemmer else word), False))
        elif m.group("cfr_num") is not None:
            # "4 CFR 25.14" names the same section as "§ 25.14"; also keep the
            # title-qualified form so cross-title references stay distinct.
            cite(f"{m.group('cfr_title')}cfr{m.group('cfr_num')}")
            cite(*_section_tokens(m.group("cfr_num"), m.group("cfr_sub")))
        elif m.group("usc_num") is not None:
            # Statute cites are a different namespace from CFR sections: "29
            # U.S.C. 206" must not collapse onto "§ 206".
            base = f"{m.group('usc_title')}usc{m.group('usc_num')}"
            sub = m.group("usc_sub")
            cite(base + sub if sub else base)
            if sub:
                cite(base)
        elif m.group("publ") is not None:
            cite("publ" + m.group("publ").replace("–", "-"))
        elif m.group("range_lo") is not None:
            cite(*_range_tokens(m.group("range_lo"), m.group("range_hi")))
        elif m.group("sec_num") is not None:
            cite(*_section_tokens(m.group("sec_num"), m.group("sec_sub")))
        elif m.group("bare_num") is not None:
            # A bare "25.14" in this corpus is almost always a section
            # reference; emit both readings and let BM25 weigh them.
            cite(m.group("bare_num"))
            cite(*_section_tokens(m.group("bare_num"), m.group("bare_sub")))
        else:  # pragma: no cover - every alternative is handled above
            raise AssertionError(f"unhandled token group {kind!r}")
    return tokens


def tokenize(text: str, *, stemmer=stem) -> list[str]:
    """Tokenize ``text`` for BM25, shared by index build and query time.

    Citations are kept whole and normalized; every other token is lowercased,
    dropped if a stopword, and stemmed. ``stemmer`` is injectable so
    alternatives can be measured against the gold set; pass ``None`` to
    disable stemming.
    """
    return [token for token, _ in _scan(text, stemmer)]


def citation_tokens(text: str) -> set[str]:
    """Just the citation tokens in ``text`` — the section references it names.

    Lets a caller tell "§ 25.14" (a reference to a specific section) apart from
    "weapons" (a topic), which is the difference between a known-item lookup
    and a search.
    """
    return {token for token, is_citation in _scan(text, None) if is_citation}
