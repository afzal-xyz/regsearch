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
            from sentence_transformers import SentenceTransformer

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
