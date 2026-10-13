"""
db.py — SQLite schema for MagiMatch v5.

Schema is intentionally lean — only columns we actually use.
Enrichment columns (props, style_summary) are nullable so the
enrichment script can fill them in a separate pass.
"""

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ── Books ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS books (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    archive_id  INTEGER UNIQUE NOT NULL,
    title       TEXT    NOT NULL,
    pub_year    TEXT,
    publisher   TEXT,
    page_count  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_books_archive_id ON books(archive_id);

-- ── Persons ────────────────────────────────────────────────────────────
-- One row per unique magician/author/subject.
-- style_summary is filled by the enrichment script.
CREATE TABLE IF NOT EXISTS persons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    UNIQUE NOT NULL,   -- display name (title-cased)
    trick_count     INTEGER DEFAULT 0,         -- updated by processor
    style_summary   TEXT                       -- filled by enrichment script
);

CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);

-- ── Book ↔ Person links ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS book_persons (
    book_id     INTEGER NOT NULL REFERENCES books(id)   ON DELETE CASCADE,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL CHECK(role IN ('author', 'subject')),
    PRIMARY KEY (book_id, person_id, role)
);

CREATE INDEX IF NOT EXISTS idx_book_persons_book   ON book_persons(book_id);
CREATE INDEX IF NOT EXISTS idx_book_persons_person ON book_persons(person_id);

-- ── Tricks ─────────────────────────────────────────────────────────────
-- effect_category = deepest category leaf from the archive breadcrumb
-- props           = comma-separated top-level prop tags (filled by enrichment)
-- credited_to     = comma-separated creator names exactly as scraped
-- embed_text      = pre-built text for the embedding model
CREATE TABLE IF NOT EXISTS tricks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id         INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    title           TEXT    NOT NULL,
    page_number     TEXT,
    description     TEXT,               -- raw archive notes, may be null
    effect_category TEXT,               -- deepest category leaf
    props           TEXT,               -- enrichment: "cards, coins" etc.
    credited_to     TEXT,               -- "Dai Vernon, Ed Marlo" etc.
    inventor_id     INTEGER REFERENCES persons(id),  -- primary creator FK
    embed_text      TEXT                -- built by processor
);

CREATE INDEX IF NOT EXISTS idx_tricks_book_id       ON tricks(book_id);
CREATE INDEX IF NOT EXISTS idx_tricks_inventor_id   ON tricks(inventor_id);
CREATE INDEX IF NOT EXISTS idx_tricks_effect_category ON tricks(effect_category);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    """WAL mode, foreign keys on, row_factory for dict-style access."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(db_path: Path):
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
