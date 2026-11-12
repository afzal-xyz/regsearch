"""Embed contextualized chunks with sentence-transformers.

Default model: ``BAAI/bge-small-en-v1.5`` (384-dim), configurable.

The embedding INPUT is the *contextualized* form of a chunk — its heading path
joined with " > ", a newline, then the chunk text — so the vector captures where
the passage sits in the CFR hierarchy. The chunk's stored ``text`` stays PURE;
contextualization happens only here, at embed/query time.

``sentence_transformers`` (and torch) are imported lazily, so importing this
module — and tests that inject a fake embedder — never pay the torch import cost.
"""

from __future__ import annotations

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# bge retrieval models expect a short instruction on the QUERY side only;
# passages are embedded as-is. Improves short-query retrieval quality.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _heading_path(chunk) -> list[str]:
    hp = chunk["heading_path"] if isinstance(chunk, dict) else chunk.heading_path
    if isinstance(hp, str):  # stored as JSON in the db
        try:
            hp = json.loads(hp)
        except json.JSONDecodeError:
            hp = [hp]
    return list(hp)


def contextualize(chunk) -> str:
    """Return the embedding input: ``heading_path`` joined by " > ", newline, text."""
    text = chunk["text"] if isinstance(chunk, dict) else chunk.text
    prefix = " > ".join(_heading_path(chunk))
    return f"{prefix}\n{text}" if prefix else text


def import_sentence_transformers():
    """Import ``sentence_transformers``, explaining OS code-integrity blocks.

    On Windows with Smart App Control (or a WDAC policy) enforcing, a compiled
    extension deep in the import chain (sentence_transformers -> sklearn ->
    scipy) can be refused with "DLL load failed ... An Application Control
    policy has blocked this file". That is an OS policy decision, not a broken
    install — reinstalling yields the same bytes and the same block — so say
    so plainly instead of surfacing a 40-frame traceback.
    """
    try:
        import sentence_transformers
    except ImportError as e:
        msg = str(e)
        if "Application Control" in msg or "DLL load failed" in msg:
            raise ImportError(
                "the ML stack could not be loaded: Windows refused a compiled "
                f"extension ({msg}).\n"
                "This is Smart App Control / App Control for Business blocking an "
                "unsigned binary, not a RegSearch or install bug; reinstalling will "
                "not help. Options:\n"
                "  - run model-backed commands (prescore, tune, onboard, eval) from "
                "WSL or Linux, where the policy does not apply;\n"
                "  - on a managed machine, ask IT to allow the file;\n"
                "  - on a personal machine, turning Smart App Control off is the "
                "only local switch (Windows Security > App & browser control).\n"
                "Blocked files are listed in Event Viewer > Applications and "
                "Services Logs > Microsoft > Windows > CodeIntegrity > Operational."
            ) from e
        raise
    return sentence_transformers


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self._device = device
        self.query_instruction = query_instruction
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        if self._model is None:
            SentenceTransformer = import_sentence_transformers().SentenceTransformer

            logger.info("loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    @property
    def dim(self) -> int:
        # get_sentence_embedding_dimension was renamed to get_embedding_dimension.
        getter = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(getter())

    def encode(self, texts, *, batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Encode passages (already contextualized). Returns float32 [N, dim]."""
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query, prepending the bge query instruction."""
        text = f"{self.query_instruction}{query}" if self.query_instruction else query
        return self.encode([text], show_progress=False)[0]
