"""Deterministically sample sections from a title's chunk set, stratified across
CFR parts, as candidates for drafting gold-standard eval questions.

The seed is fixed and echoed in the output so the gold set's provenance is fully
reproducible: the same seed + same chunk file always yields the same sample.

Usage:
    python -m eval.sample_sections [--chunks data/chunks/title-4-2024-07-18.jsonl]
                                   [--n 45] [--seed 20260717]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_SEED = 20260717
DEFAULT_CHUNKS = Path("data/chunks/title-4-2024-07-18.jsonl")


def _part_of(section_citation: str) -> str:
    m = re.search(r"§\s*(\d+)", section_citation)
    if m:
        return m.group(1)
    m = re.search(r"Part\s+(\d+)", section_citation)
    return m.group(1) if m else "?"


def _part_key(part: str) -> int:
    return int(part) if part.isdigit() else 10_000


def load_sections(chunks_path: Path | str) -> list[dict]:
    """Group chunks back into sections (body = chunk texts joined in order)."""
    lines = Path(chunks_path).read_text(encoding="utf-8").splitlines()
    by_section: dict[str, dict] = {}
    for line in lines:
        if not line.strip():
            continue
        c = json.loads(line)
        sc = c["section_citation"]
        sec = by_section.setdefault(
            sc,
            {
                "section_citation": sc,
                "heading": c["heading"],
                "heading_path": c["heading_path"],
                "part": _part_of(sc),
                "_pieces": [],
            },
        )
        sec["_pieces"].append((c["char_start"], c["text"]))
    sections = []
    for sec in by_section.values():
        sec["body"] = "\n\n".join(text for _, text in sorted(sec["_pieces"]))
        del sec["_pieces"]
        sections.append(sec)
    return sections


def _allocate(parts: list[str], sizes: dict[str, int], n: int, total: int) -> dict[str, int]:
    """Floor of 1 per part, remainder distributed by largest fractional share."""
    alloc = {p: 1 for p in parts}
    remaining = n - len(parts)
    if remaining <= 0:
        return alloc
    shares = {p: remaining * sizes[p] / total for p in parts}
    floors = {p: int(shares[p]) for p in parts}
    for p in parts:
        alloc[p] += floors[p]
    leftover = remaining - sum(floors.values())
    by_frac = sorted(parts, key=lambda p: shares[p] - floors[p], reverse=True)
    for p in by_frac[:leftover]:
        alloc[p] += 1
    return alloc


def stratified_sample(sections: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_part: dict[str, list[dict]] = defaultdict(list)
    for s in sections:
        by_part[s["part"]].append(s)
    parts = sorted(by_part, key=_part_key)
    sizes = {p: len(by_part[p]) for p in parts}
    alloc = _allocate(parts, sizes, n, len(sections))

    sampled: list[dict] = []
    for p in parts:
        pool = sorted(by_part[p], key=lambda s: s["section_citation"])
        sampled.extend(rng.sample(pool, min(alloc[p], len(pool))))
    sampled.sort(key=lambda s: (_part_key(s["part"]), s["section_citation"]))
    return sampled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.sample_sections")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--n", type=int, default=45)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    sections = load_sections(args.chunks)
    sampled = stratified_sample(sections, args.n, args.seed)
    out = {
        "seed": args.seed,
        "source": str(args.chunks),
        "total_sections": len(sections),
        "sampled": len(sampled),
        "sections": sampled,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
