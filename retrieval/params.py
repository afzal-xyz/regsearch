"""Tunable retrieval parameters, stored with the provenance that justifies them.

The fusion constants in :mod:`retrieval.search` were calibrated by hand
against a particular corpus and a particular gold set, and the code comments
said so — "re-tune when the corpus or gold set grows". A comment is not a
mechanism. When Title 1 was onboarded the corpus grew, nothing re-tuned, and
hybrid's prose MRR slipped below vector-only's without any stage noticing.

So the parameters now live in ``retrieval/params.json`` next to a record of
what they were tuned on:

    {
      "params": {"rrf_k": 10, "bm25_weight": 1.0, "bm25_contrast_lo": 1.4, ...},
      "provenance": {"corpus": ["1@2024-07-18", ...], "gold_sha256": "...",
                     "analyzer_version": 5, "tuned_at": "...", "objective": 0.81}
    }

The integrity gate compares that provenance with the live corpus and warns
when they disagree; :mod:`retrieval.tune` is what brings them back into
agreement, and :mod:`pipeline.onboard` runs it before every eval. The file is
committed, so a checkout always carries parameters that match its corpus.

When the file is absent the hand-tuned defaults from the code apply, so
nothing here changes behaviour until the first tune is run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PARAMS_PATH = Path("retrieval/params.json")


@dataclass(frozen=True)
class FusionParams:
    """Everything :func:`retrieval.search.fuse` needs besides the candidates."""

    rrf_k: int = 10
    bm25_weight: float = 1.0
    bm25_contrast_anchor: int = 20
    bm25_contrast_lo: float = 1.4
    bm25_contrast_hi: float = 2.2
    # Agreement gate. BM25's contrast-based confidence is additionally scaled
    # by how much its top-`agreement_k` sections overlap vector's: an overlap
    # of `agreement_scale` (as a fraction of k) or more leaves confidence
    # untouched, less scales it down linearly, none zeroes it. 0 disables the
    # gate (the pre-Title-1 behaviour). Never applied when the query names an
    # indexed section — a citation is a lexical anchor, and the two retrievers
    # *should* disagree there, because vector cannot see citations.
    agreement_scale: float = 0.0
    agreement_k: int = 10
    # Sections hybrid hands to the cross-encoder. Lives here rather than in
    # rag/ because the sweep that picks it is the same sweep that picks the
    # fusion constants — they are tuned together, against the same gold set.
    rerank_candidates: int = 10

    def with_(self, **changes) -> FusionParams:
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FusionParams:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


DEFAULTS = FusionParams()


@dataclass(frozen=True)
class Provenance:
    corpus: tuple[str, ...]  # manifest.corpus_state(...) at tuning time
    gold_sha256: str  # eval.run.gold_fingerprint(...)["sha256"]
    gold_n: int
    analyzer_version: int
    tuned_at: str
    objective: float | None = None  # cross-validated score of these params
    note: str = ""
    # The cross-encoder these parameters were tuned with. rerank_candidates in
    # particular is a property of the reranker: a stronger model may earn a
    # wider pool, a weaker one is hurt by it. Swapping the model therefore
    # makes the params stale, like growing the corpus does.
    rerank_model: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["corpus"] = list(self.corpus)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Provenance:
        return cls(
            corpus=tuple(d.get("corpus", ())),
            gold_sha256=str(d.get("gold_sha256", "")),
            gold_n=int(d.get("gold_n", 0)),
            analyzer_version=int(d.get("analyzer_version", 0)),
            tuned_at=str(d.get("tuned_at", "")),
            objective=d.get("objective"),
            note=str(d.get("note", "")),
            rerank_model=str(d.get("rerank_model", "")),
        )


def load(path: Path | str = DEFAULT_PARAMS_PATH) -> tuple[FusionParams, Provenance | None]:
    """Parameters from ``path``; the code defaults (and no provenance) if absent."""
    path = Path(path)
    if not path.exists():
        return DEFAULTS, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        params = FusionParams.from_dict(data.get("params", {}))
        prov = Provenance.from_dict(data["provenance"]) if data.get("provenance") else None
        return params, prov
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A corrupt params file must not take search down with it: fall back to
        # the defaults loudly rather than fail every query.
        logger.warning("could not read %s (%s); using default fusion params", path, exc)
        return DEFAULTS, None


def save(
    params: FusionParams,
    provenance: Provenance,
    path: Path | str = DEFAULT_PARAMS_PATH,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": params.to_dict(), "provenance": provenance.to_dict()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Process-wide cache so hybrid_search does not re-read the file per query. The
# tuner and tests call `reset()` after writing a new file.
_CACHE: dict[str, tuple[FusionParams, Provenance | None]] = {}


def current(path: Path | str = DEFAULT_PARAMS_PATH) -> FusionParams:
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = load(path)
    return _CACHE[key][0]


def reset() -> None:
    _CACHE.clear()
