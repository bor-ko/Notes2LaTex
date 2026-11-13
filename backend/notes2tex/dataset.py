"""SQLite store for documents, pages and blocks.

Every block keeps three versions of its LaTeX:
  draft_latex  what the model produced (never edited by hand)
  latex        the current working version in the review UI
  final_latex  set when you approve the page: this is the training label
The difference between draft and final is what the accuracy metric measures.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    subject TEXT NOT NULL,
    source_path TEXT,
    split TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    mask_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',      -- new | transcribing | transcribed | reviewed | error
    model TEXT,
    usage_json TEXT,
    error TEXT,
    transcribed_at TEXT,
    reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS blocks (
    id INTEGER PRIMARY KEY,
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    x0 INTEGER NOT NULL, y0 INTEGER NOT NULL, x1 INTEGER NOT NULL, y1 INTEGER NOT NULL,
    crop_path TEXT,
    draft_role TEXT,
    draft_latex TEXT,
    role TEXT NOT NULL DEFAULT 'prose',
    latex TEXT NOT NULL DEFAULT '',
    continues_previous INTEGER NOT NULL DEFAULT 0,
    code_language TEXT,
    final_latex TEXT,
    final_role TEXT,
    source_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_pages_doc ON pages(document_id);
CREATE INDEX IF NOT EXISTS idx_blocks_page ON blocks(page_id);
"""


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def assign_split(name: str, doc_id: int) -> str:
    """Deterministic split by document, so no page of a held-out document
    ever ends up in training. Roughly 75 / 10 / 15."""
    h = int(hashlib.sha256(f"{doc_id}:{name}".encode()).hexdigest(), 16) % 100
    if h < 15:
        return "test"
    if h < 25:
        return "val"
    return "train"


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(r: sqlite3.Row | None) -> dict | None:
    return dict(r) if r is not None else None


# ---- documents -------------------------------------------------------------


def create_document(conn: sqlite3.Connection, name: str, subject: str, source_path: str | None) -> int:
    cur = conn.execute(
        "INSERT INTO documents (name, subject, source_path, split, created_at) VALUES (?, ?, ?, 'pending', ?)",
        (name, subject, source_path, now()),
    )
    doc_id = cur.lastrowid
    conn.execute("UPDATE documents SET split = ? WHERE id = ?", (assign_split(name, doc_id), doc_id))
    return doc_id


def get_document(conn: sqlite3.Connection, doc_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone())


def list_documents(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT d.*,
               COUNT(p.id) AS page_count,
               SUM(p.status = 'reviewed') AS reviewed_count,
               SUM(p.status = 'transcribed') AS transcribed_count,
               SUM(p.status IN ('new', 'transcribing')) AS pending_count,
               SUM(p.status = 'error') AS error_count
        FROM documents d LEFT JOIN pages p ON p.document_id = d.id
        GROUP BY d.id ORDER BY d.created_at DESC, d.id DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def delete_document(conn: sqlite3.Connection, doc_id: int) -> None:
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


# ---- pages -----------------------------------------------------------------


def add_page(conn: sqlite3.Connection, doc_id: int, index: int, image_path: str, mask_path: str, width: int, height: int) -> int:
    cur = conn.execute(
        "INSERT INTO pages (document_id, page_index, image_path, mask_path, width, height) VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, index, image_path, mask_path, width, height),
    )
    return cur.lastrowid


def get_page(conn: sqlite3.Connection, page_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone())


def list_pages(conn: sqlite3.Connection, doc_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT p.*, COUNT(b.id) AS block_count,
               SUM(instr(b.latex, '\\unsure{') > 0) AS unsure_count
        FROM pages p LEFT JOIN blocks b ON b.page_id = p.id
        WHERE p.document_id = ? GROUP BY p.id ORDER BY p.page_index
        """,
        (doc_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def set_page_status(conn: sqlite3.Connection, page_id: int, status: str, error: str | None = None) -> None:
    conn.execute("UPDATE pages SET status = ?, error = ? WHERE id = ?", (status, error, page_id))


# ---- blocks ----------------------------------------------------------------


def replace_blocks(conn: sqlite3.Connection, page_id: int, boxes: list[tuple[int, int, int, int]]) -> list[int]:
    conn.execute("DELETE FROM blocks WHERE page_id = ?", (page_id,))
    ids = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cur = conn.execute(
            "INSERT INTO blocks (page_id, ord, x0, y0, x1, y1) VALUES (?, ?, ?, ?, ?, ?)",
            (page_id, i, x0, y0, x1, y1),
        )
        ids.append(cur.lastrowid)
    return ids


def get_blocks(conn: sqlite3.Connection, page_id: int) -> list[dict]:
    rows = conn.execute("SELECT * FROM blocks WHERE page_id = ? ORDER BY ord", (page_id,)).fetchall()
    return [dict(r) for r in rows]


def get_block(conn: sqlite3.Connection, block_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM blocks WHERE id = ?", (block_id,)).fetchone())


def set_crop_path(conn: sqlite3.Connection, block_id: int, path: str) -> None:
    conn.execute("UPDATE blocks SET crop_path = ? WHERE id = ?", (path, block_id))


def save_transcription(conn: sqlite3.Connection, page_id: int, blocks: list, model: str, usage: dict) -> None:
    """Store model output. `blocks` carry ord-based ids (1..n)."""
    rows = get_blocks(conn, page_id)
    for row, b in zip(rows, blocks):
        conn.execute(
            """UPDATE blocks SET draft_role = ?, draft_latex = ?, role = ?, latex = ?,
               continues_previous = ?, code_language = ?, source_model = ?,
               final_latex = NULL, final_role = NULL WHERE id = ?""",
            (b.role, b.latex, b.role, b.latex, int(b.continues_previous), b.code_language, model, row["id"]),
        )
    conn.execute(
        "UPDATE pages SET status = 'transcribed', model = ?, usage_json = ?, error = NULL, transcribed_at = ?, reviewed_at = NULL WHERE id = ?",
        (model, json.dumps(usage), now(), page_id),
    )


def update_block(conn: sqlite3.Connection, block_id: int, **fields) -> None:
    allowed = {"role", "latex", "continues_previous", "code_language"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if "continues_previous" in sets:
        sets["continues_previous"] = int(bool(sets["continues_previous"]))
    if not sets:
        return
    cols = ", ".join(f"{k} = ?" for k in sets)
    conn.execute(f"UPDATE blocks SET {cols} WHERE id = ?", (*sets.values(), block_id))


def apply_autofix(conn: sqlite3.Connection, block_id: int, latex: str) -> None:
    """A compile fix is still model output, so it updates the draft too
    (unless the page was already reviewed by hand)."""
    conn.execute(
        """UPDATE blocks SET latex = ?,
           draft_latex = CASE WHEN final_latex IS NULL THEN ? ELSE draft_latex END
           WHERE id = ?""",
        (latex, latex, block_id),
    )


def approve_page(conn: sqlite3.Connection, page_id: int) -> None:
    conn.execute("UPDATE blocks SET final_latex = latex, final_role = role WHERE page_id = ?", (page_id,))
    conn.execute("UPDATE pages SET status = 'reviewed', reviewed_at = ? WHERE id = ?", (now(), page_id))


def reviewed_blocks(conn: sqlite3.Connection, split: str | None = None) -> list[dict]:
    """Blocks on approved pages: (model draft, your final) pairs."""
    q = """
        SELECT b.*, p.page_index, p.image_path, d.id AS document_id, d.name AS document_name,
               d.subject, d.split
        FROM blocks b JOIN pages p ON p.id = b.page_id JOIN documents d ON d.id = p.document_id
        WHERE p.status = 'reviewed' AND b.final_latex IS NOT NULL
    """
    args: tuple = ()
    if split:
        q += " AND d.split = ?"
        args = (split,)
    return [dict(r) for r in conn.execute(q + " ORDER BY d.id, p.page_index, b.ord", args).fetchall()]
