"""Tests for the index layer (index.store) using a deterministic fake embedder.

The fake embedder is a bag-of-words model over a tiny fixed vocabulary, so it
needs no download yet still produces meaningful nearest-neighbour behaviour and
exercises the real sqlite-vec KNN path.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from index import store
from index.store import ModelMismatchError

_TOK = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Bag-of-words embedder over a fixed vocab; unit-normalized vectors."""

    VOCAB = [
        "financial", "disclosure", "gao", "employee", "file", "report",
        "travel", "reimbursement", "expense", "ethics", "conflict", "personnel",
    ]

    def __init__(self, model_name: str = "fake-bow-v1"):
        self.model_name = model_name

    @property
    def dim(self) -> int:
        return len(self.VOCAB)

    def _vec(self, text: str) -> np.ndarray:
        toks = _TOK.findall(text.lower())
        v = np.array([toks.count(w) for w in self.VOCAB], dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        else:
            v[0] = 1.0  # avoid an all-zero vector
        return v

    def encode(self, texts, *, batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, query: str) -> np.ndarray:
        return self._vec(query)


def _chunk(chunk_id, citation, heading, text, *, title=4, date="2024-07-18", anchor=None):
    return {
        "chunk_id": chunk_id,
        "citation": citation,
        "section_citation": citation.split("(")[0],
        "heading": heading,
        "heading_path": ["Title 4—Accounts", heading],
        "title": title,
        "date": date,
        "paragraph_anchor": anchor,
        "char_start": 0,
        "char_end": len(text),
        "token_estimate": len(text) // 4,
        "text": text,
    }


FIXTURE_CHUNKS = [
    _chunk(
        "4-2024-07-18-7.3-000", "4 CFR § 7.3", "§ 7.3 Financial disclosure.",
        "A GAO employee must file a financial disclosure report annually and upon "
        "termination of employment.",
    ),
    _chunk(
        "4-2024-07-18-8.1-000", "4 CFR § 8.1", "§ 8.1 Travel reimbursement.",
        "An employee may claim travel reimbursement for expense incurred on "
        "official business.",
    ),
    _chunk(
        "4-2024-07-18-2.4-000", "4 CFR § 2.4", "§ 2.4 Merit system principles.",
        "Personnel decisions shall be based on merit and free of conflict of "
        "interest or personal favoritism.",
    ),
    _chunk(
        "4-2024-07-18-9.2-000", "4 CFR § 9.2", "§ 9.2 Ethics counseling.",
        "The ethics official provides counseling to any employee on conflict of "
        "interest questions.",
    ),
]


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "index" / "regsearch.db"


@pytest.fixture
def built(db_path):
    emb = FakeEmbedder()
    summary = store.build_index(FIXTURE_CHUNKS, db_path=db_path, embedder=emb, show_progress=False)
    return db_path, emb, summary


def test_build_populates_chunks_vectors_bm25_meta(built):
    db_path, emb, summary = built
    assert summary["added"] == len(FIXTURE_CHUNKS)
    conn = store.connect(db_path)
    try:
        assert conn.execute("select count(*) from chunks").fetchone()[0] == 4
        assert conn.execute("select count(*) from vec_chunks").fetchone()[0] == 4
        assert conn.execute("select count(*) from bm25").fetchone()[0] == 1
        assert store.get_model_name(conn) == "fake-bow-v1"
        assert int(store.get_meta(conn)["dim"]) == emb.dim
    finally:
        conn.close()


def test_stored_text_is_pure(built):
    db_path, _, _ = built
    conn = store.connect(db_path)
    try:
        row = conn.execute(
            "select text from chunks where chunk_id = ?", ("4-2024-07-18-7.3-000",)
        ).fetchone()
        assert row["text"] == FIXTURE_CHUNKS[0]["text"]
        # no heading path baked into stored text
        assert "Title 4" not in row["text"]
    finally:
        conn.close()


def test_idempotent_per_title_date(built):
    db_path, emb, _ = built
    # Rebuild the same title/date; counts must stay stable and ids unique.
    store.build_index(FIXTURE_CHUNKS, db_path=db_path, embedder=emb, show_progress=False)
    conn = store.connect(db_path)
    try:
        assert conn.execute("select count(*) from chunks").fetchone()[0] == 4
        assert conn.execute("select count(*) from vec_chunks").fetchone()[0] == 4
        distinct = conn.execute("select count(distinct chunk_id) from chunks").fetchone()[0]
        assert distinct == 4
    finally:
        conn.close()


def test_model_mismatch_at_build_fails_loudly(built):
    db_path, _, _ = built
    with pytest.raises(ModelMismatchError):
        store.build_index(
            FIXTURE_CHUNKS,
            db_path=db_path,
            embedder=FakeEmbedder("other-model"),
            show_progress=False,
        )


@pytest.mark.slow
def test_real_model_dim():
    from index.embedder import Embedder

    emb = Embedder()
    assert emb.dim == 384
    vecs = emb.encode(["a test sentence"], show_progress=False)
    assert vecs.shape == (1, 384)
