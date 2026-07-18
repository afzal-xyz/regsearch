"""SQLite-backed index store: chunk metadata + pure text, vector embeddings
(sqlite-vec), and a persisted BM25 index (rank-bm25).

Everything lives in a single-file database (default ``data/index/regsearch.db``):

- ``chunks``     — all chunk metadata plus the PURE chunk text.
- ``vec_chunks`` — a sqlite-vec ``vec0`` virtual table, one row per chunk
                   (``rowid == chunks.id``), holding the normalized embedding.
- ``bm25``       — the pickled BM25 index (see :func:`load_bm25`).
- ``meta``       — model name / embedding dim, used to guard against mixing
                   incompatible embeddings.

``build_index`` is idempotent per ``(title, date)``: re-running replaces that
title/date's rows and rebuilds the (global) BM25 index.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import sqlite_vec
from rank_bm25 import BM25Okapi

from index.embedder import DEFAULT_MODEL, contextualize

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/index/regsearch.db")

_CHUNK_COLUMNS = [
    "chunk_id",
    "citation",
    "section_citation",
    "heading",
    "heading_path",
    "title",
    "date",
    "paragraph_anchor",
    "char_start",
    "char_end",
    "token_estimate",
    "text",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class ModelMismatchError(RuntimeError):
    """Raised when an embedder's model does not match the index's meta table."""


def tokenize(text: str) -> list[str]:
    """Lowercase ``[a-z0-9]+`` tokenizer, shared by BM25 build and query time."""
    return _TOKEN_RE.findall(text.lower())


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def get_meta(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute("select key, value from meta").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r["key"]: r["value"] for r in rows}


def get_model_name(conn: sqlite3.Connection) -> str | None:
    return get_meta(conn).get("model_name")


def assert_model_matches(conn: sqlite3.Connection, embedder) -> None:
    """Fail loudly if ``embedder`` was not the model used to build the index.

    Querying with a different model produces vectors in an incompatible space
    (and possibly the wrong dimension), so silently returning garbage would be
    worse than erroring.
    """
    stored = get_model_name(conn)
    if stored is not None and stored != embedder.model_name:
        raise ModelMismatchError(
            f"query embedder model {embedder.model_name!r} does not match "
            f"index model {stored!r}; rebuild the index or use the matching model"
        )


def _init_schema(conn: sqlite3.Connection, dim: int) -> None:
    conn.execute(
        """
        create table if not exists chunks (
            id integer primary key,
            chunk_id text unique not null,
            citation text,
            section_citation text,
            heading text,
            heading_path text,
            title integer,
            date text,
            paragraph_anchor text,
            char_start integer,
            char_end integer,
            token_estimate integer,
            text text
        )
        """
    )
    conn.execute("create index if not exists idx_chunks_title_date on chunks(title, date)")
    conn.execute(
        f"create virtual table if not exists vec_chunks using vec0(embedding float[{dim}])"
    )
    conn.execute(
        "create table if not exists bm25 (id integer primary key check (id = 1), data blob)"
    )
    conn.execute("create table if not exists meta (key text primary key, value text)")


def _validate_or_set_meta(conn: sqlite3.Connection, model_name: str, dim: int) -> None:
    meta = get_meta(conn)
    if not meta.get("model_name"):
        conn.executemany(
            "insert or replace into meta(key, value) values (?, ?)",
            [
                ("model_name", model_name),
                ("dim", str(dim)),
                ("created_at", str(time.time())),
            ],
        )
        return
    if meta["model_name"] != model_name:
        raise ModelMismatchError(
            f"index was built with model {meta['model_name']!r}; "
            f"refusing to add chunks embedded with {model_name!r}"
        )
    if int(meta["dim"]) != dim:
        raise ModelMismatchError(f"index dim {meta['dim']} != embedder dim {dim}")


def _load_jsonl(source: Path | str | list[dict]) -> list[dict]:
    if isinstance(source, list):
        return source
    chunks: list[dict] = []
    with open(source, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _delete_title_date(conn: sqlite3.Connection, title: int, date: str) -> None:
    rows = conn.execute("select id from chunks where title = ? and date = ?", (title, date))
    ids = [r["id"] for r in rows]
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    conn.executemany("delete from vec_chunks where rowid = ?", [(i,) for i in ids])
    conn.execute(f"delete from chunks where id in ({placeholders})", ids)


def _insert_chunk(conn: sqlite3.Connection, chunk: dict, vector: np.ndarray) -> None:
    heading_path = chunk["heading_path"]
    if isinstance(heading_path, list):
        heading_path = json.dumps(heading_path, ensure_ascii=False)
    values = [chunk.get(col) if col != "heading_path" else heading_path for col in _CHUNK_COLUMNS]
    placeholders = ",".join("?" * len(_CHUNK_COLUMNS))
    cur = conn.execute(
        f"insert into chunks ({','.join(_CHUNK_COLUMNS)}) values ({placeholders})", values
    )
    conn.execute(
        "insert into vec_chunks(rowid, embedding) values (?, ?)",
        (cur.lastrowid, sqlite_vec.serialize_float32(vector.tolist())),
    )


def _rebuild_bm25(conn: sqlite3.Connection) -> None:
    rows = conn.execute("select chunk_id, text from chunks order by id").fetchall()
    conn.execute("delete from bm25")
    if not rows:
        return
    ids = [r["chunk_id"] for r in rows]
    model = BM25Okapi([tokenize(r["text"]) for r in rows])
    blob = pickle.dumps({"ids": ids, "model": model})
    conn.execute("insert into bm25(id, data) values (1, ?)", (blob,))


def load_bm25(conn: sqlite3.Connection):
    """Load the persisted BM25 index as ``(ids, BM25Okapi | None)``.

    The index is pickled into the ``bm25`` blob. Pickle is safe here because the
    database is a LOCAL, TRUSTED artifact that RegSearch builds itself — it is
    never sourced from an untrusted party, so pickle's arbitrary-code-execution
    risk does not apply. If the blob is missing or unreadable (e.g. a rank-bm25
    upgrade changed the pickle layout), we transparently rebuild from
    ``chunks.text`` so search still works.
    """
    row = conn.execute("select data from bm25 where id = 1").fetchone()
    if row and row["data"]:
        try:
            payload = pickle.loads(row["data"])
            return payload["ids"], payload["model"]
        except Exception:  # noqa: BLE001 - any unpickle failure -> rebuild
            logger.warning("bm25 blob unreadable; rebuilding from chunks.text")
    rows = conn.execute("select chunk_id, text from chunks order by id").fetchall()
    ids = [r["chunk_id"] for r in rows]
    model = BM25Okapi([tokenize(r["text"]) for r in rows]) if rows else None
    return ids, model


def build_index(
    chunks_jsonl: Path | str | list[dict],
    *,
    db_path: Path | str = DEFAULT_DB,
    embedder=None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
) -> dict:
    """Build (or update) the index from a chunk JSONL file. Idempotent per
    ``(title, date)``. Returns a small summary dict."""
    chunks = _load_jsonl(chunks_jsonl)
    if not chunks:
        raise ValueError(f"no chunks found in {chunks_jsonl!r}")

    if embedder is None:
        from index.embedder import Embedder

        embedder = Embedder(model_name)
    dim = embedder.dim

    conn = connect(db_path)
    try:
        _init_schema(conn, dim)
        _validate_or_set_meta(conn, embedder.model_name, dim)

        title_dates = sorted({(c["title"], c["date"]) for c in chunks})
        for title, date in title_dates:
            _delete_title_date(conn, title, date)

        vectors = embedder.encode(
            [contextualize(c) for c in chunks],
            batch_size=batch_size,
            show_progress=show_progress,
        )
        for chunk, vector in zip(chunks, vectors, strict=True):
            _insert_chunk(conn, chunk, np.asarray(vector, dtype=np.float32))

        _rebuild_bm25(conn)
        conn.commit()

        total = conn.execute("select count(*) from chunks").fetchone()[0]
        logger.info("indexed %d chunks (%d total) for %s", len(chunks), total, title_dates)
        return {"added": len(chunks), "total": total, "title_dates": title_dates}
    finally:
        conn.close()
