"""Parse cached eCFR full-title XML into a tree of :class:`Section` records.

eCFR XML nests ``<DIVn>`` elements, but the depth number ``n`` is NOT reliable
(a title uses DIV1, DIV3, DIV4, DIV5, DIV6, DIV7, DIV8, DIV9). The ``TYPE``
attribute is the source of truth:

    TITLE > CHAPTER > SUBCHAP > PART > SUBPART > SUBJGRP > SECTION | APPENDIX

Container types contribute to a section's ``heading_path``; ``SECTION`` and
``APPENDIX`` are leaves that become :class:`Section` records.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

CONTAINER_TYPES = {"TITLE", "CHAPTER", "SUBCHAP", "PART", "SUBPART", "SUBJGRP"}
LEAF_TYPES = {"SECTION", "APPENDIX"}

# Direct children of a leaf DIV: which carry body text, which are deliberately
# ignored. Anything else triggers a warning (see _collect_body).
_BODY_TAGS = {"P", "FP", "HD2", "HD3"}
_BLOCK_CONTAINER_TAGS = {"EXTRACT"}  # wraps nested body paragraphs
_SKIP_TAGS = {"HEAD", "AUTH", "SOURCE", "CITA", "EDNOTE"}

_DIV_RE = re.compile(r"DIV\d+$")


def _norm(text: str | None) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return " ".join((text or "").split())


@dataclass
class Section:
    title: int  # source CFR title number
    date: str  # source issue date (YYYY-MM-DD)
    number: str  # DIV "N" attribute, e.g. "2.1" or "Appendix I to Part 83"
    citation: str  # e.g. "4 CFR § 2.1"
    heading: str  # section heading, normalized
    heading_path: list[str]  # ancestor headings, title -> section (own heading last)
    paragraphs: list[str] = field(default_factory=list)  # body, order preserved

    def __post_init__(self) -> None:
        # Invariant: the section's own heading is the tail of its path.
        if not self.heading_path or self.heading_path[-1] != self.heading:
            raise ValueError(
                f"heading_path[-1] must equal heading for {self.citation!r}"
            )

    @property
    def body(self) -> str:
        """Body text with paragraph structure preserved (blank-line separated)."""
        return "\n\n".join(self.paragraphs)


def _head_text(div: ET.Element) -> str:
    head = div.find("HEAD")
    if head is None:
        return ""
    return _norm("".join(head.itertext()))


def _citation(title: int, div_type: str, number: str) -> str:
    if div_type == "SECTION":
        return f"{title} CFR § {number}"
    # APPENDIX and any other leaf: N is already descriptive ("Appendix I to Part 83").
    return f"{title} CFR {number}".rstrip()


def _collect_body(leaf: ET.Element, citation: str) -> list[str]:
    """Extract ordered body paragraphs from a SECTION/APPENDIX element.

    Warns (rather than silently dropping) on unrecognized block-level children,
    so new eCFR element types surface instead of vanishing from the corpus.
    """
    paragraphs: list[str] = []
    for child in leaf:
        tag = child.tag
        if tag in _BODY_TAGS:
            text = _norm("".join(child.itertext()))
            if text:
                paragraphs.append(text)
        elif tag in _BLOCK_CONTAINER_TAGS:
            for sub in child:
                if sub.tag in _BODY_TAGS:
                    text = _norm("".join(sub.itertext()))
                    if text:
                        paragraphs.append(text)
        elif tag in _SKIP_TAGS:
            continue
        else:
            logger.warning(
                "unrecognized block element <%s> in %s; skipping", tag, citation
            )
    return paragraphs


def _walk(
    elem: ET.Element,
    title: int,
    date: str,
    ancestors: list[str],
    out: list[Section],
) -> None:
    for child in elem:
        if not _DIV_RE.match(child.tag):
            continue
        div_type = child.get("TYPE", "")
        heading = _head_text(child)

        if div_type in LEAF_TYPES:
            number = child.get("N", "")
            citation = _citation(title, div_type, number)
            out.append(
                Section(
                    title=title,
                    date=date,
                    number=number,
                    citation=citation,
                    heading=heading,
                    heading_path=[*ancestors, heading],
                    paragraphs=_collect_body(child, citation),
                )
            )
        elif div_type in CONTAINER_TYPES:
            next_ancestors = [*ancestors, heading] if heading else ancestors
            _walk(child, title, date, next_ancestors, out)
        else:
            logger.warning("unrecognized DIV type %r; traversing children", div_type)
            next_ancestors = [*ancestors, heading] if heading else ancestors
            _walk(child, title, date, next_ancestors, out)


def _load(source: str | bytes | Path) -> ET.Element:
    if isinstance(source, (bytes, bytearray)):
        return ET.fromstring(source)
    if isinstance(source, Path):
        return ET.parse(source).getroot()
    # str: a short, newline-free existing path is a file; otherwise raw XML.
    if "\n" not in source and len(source) < 1000 and Path(source).exists():
        return ET.parse(source).getroot()
    return ET.fromstring(source)


def parse_title(source: str | bytes | Path, title: int, date: str) -> list[Section]:
    """Parse full-title eCFR XML into an ordered list of Section records.

    ``source`` may be a path, raw XML bytes, or an XML string.
    """
    root = _load(source)
    sections: list[Section] = []
    _walk(root, int(title), date, [], sections)
    return sections


def parse_file(path: str | Path, title: int, date: str) -> list[Section]:
    return parse_title(Path(path), title, date)
