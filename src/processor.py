"""
processor.py — MagiMatch v5: loads scraped JSON batches into SQLite.

Design principles:
  - Lean schema: only columns we actually use
  - SQL-level dedup and counting (no Python loops for aggregation)
  - Props extracted rule-based from category breadcrumbs (no LLM)
  - embed_text built here so it's ready for the embedder
  - Enrichment columns (style_summary) left NULL for the enrichment script

Usage:
    PYTHONPATH=. python src/processor.py
    PYTHONPATH=. python src/processor.py --resume
    PYTHONPATH=. python src/processor.py --reset
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import Checkpoint
from src.config_loader import cfg, paths
from src.db import get_connection, init_schema
from src.logger import get_logger

logger = get_logger("processor")


# ── Junk filters ───────────────────────────────────────────────────────────────

JUNK_TITLES = {t.lower() for t in cfg.processor.junk_titles}
JUNK_CATS = {c.lower() for c in cfg.processor.junk_categories}


def _is_junk_title(title: str) -> bool:
    t = title.strip().lower()
    if t in JUNK_TITLES:
        return True
    # Also catch "Introduction to the Second Edition" etc.
    for junk in JUNK_TITLES:
        if t.startswith(junk) and len(t) < len(junk) + 30:
            return True
    return False


# ── Prop extraction from category breadcrumbs ──────────────────────────────────
# The top-level category segment is the prop/domain.
# We normalize common variants to a clean set of prop tags.

PROP_MAP = {
    # Cards
    "card": "cards",
    "cards": "cards",
    # Coins
    "coin": "coins",
    "coins": "coins",
    # Rope
    "rope": "rope",
    # Rings
    "ring": "rings",
    "rings": "rings",
    # Dice
    "dice": "dice",
    # Cups
    "cup": "cups",
    "cups & balls": "cups & balls",
    # Silks
    "silk": "silks",
    "silks": "silks",
    # Mentalism / no props
    "mentalism": "mentalism",
    "mental": "mentalism",
    # Balls
    "ball": "balls",
    "balls": "balls",
    # Thimble
    "thimble": "thimble",
    # Sponge
    "sponge": "sponge",
    # Paper
    "paper": "paper",
    # Money
    "money": "money",
    "bill": "money",
    "bills": "money",
    "banknote": "money",
    # Thread
    "thread": "thread",
    # Rubber bands
    "rubber band": "rubber bands",
    "rubber bands": "rubber bands",
    # Matches
    "matches": "matches",
    "match": "matches",
    # Handkerchief
    "handkerchief": "handkerchief",
    # Apparatus / gimmick
    "apparatus & dealer props": "gimmick",
    "gaffed": "gimmick",
    # Books / papers
    "book": "book test",
    # General catch-all
    "general": None,
    "games": None,
}


def extract_props(categories: list[str]) -> str | None:
    """
    Extract prop tags from category breadcrumbs.
    Takes the top-level segment of each breadcrumb, normalizes it.
    Returns comma-separated string or None.
    """
    props = set()
    for cat_path in categories:
        segments = [s.strip() for s in cat_path.split(" / ")]
        if not segments:
            continue
        top = segments[0].lower()
        prop = PROP_MAP.get(top)
        if prop:
            props.add(prop)
        elif top not in JUNK_CATS and top not in ("scientific amusement",):
            # Keep unrecognized top-level categories as-is (title-cased)
            # They'll be useful for search even without normalization
            props.add(segments[0].title())

    if not props:
        return None
    return ", ".join(sorted(props))


# ── embed_text builder ─────────────────────────────────────────────────────────


def build_embed_text(
    trick_title: str,
    description: str | None,
    book_title: str,
    authors: list[str],
    categories: list[str],
) -> str:
    """
    Build a rich text string for semantic embedding.
    Format: "{title}. {description}. From: {book} by {authors}. [{categories}]"
    """
    parts = [f"{trick_title}."]
    if description:
        parts.append(description)
    author_str = ", ".join(authors) if authors else "Unknown"
    parts.append(f"From: {book_title} by {author_str}.")
    # Use the deepest leaves of each category path
    leaves = []
    for cat in categories:
        leaf = cat.split(" / ")[-1].strip()
        if leaf:
            leaves.append(leaf)
    if leaves:
        parts.append(f"[{', '.join(leaves)}]")
    return " ".join(parts)


# ── Person upsert ──────────────────────────────────────────────────────────────


def upsert_person(conn, name_cache: dict, name: str) -> int:
    """Return person_id, creating if new. Uses in-memory cache for speed."""
    # Normalize: title-case, collapse whitespace
    canon = re.sub(r"\s+", " ", name.strip()).title()
    if canon in name_cache:
        return name_cache[canon]
    row = conn.execute("SELECT id FROM persons WHERE name = ?", (canon,)).fetchone()
    if row:
        pid = row["id"]
    else:
        cur = conn.execute("INSERT INTO persons (name) VALUES (?)", (canon,))
        pid = cur.lastrowid
    name_cache[canon] = pid
    return pid


# ── Book loader ────────────────────────────────────────────────────────────────


def load_book(book_data: dict, conn, name_cache: dict) -> int:
    """
    Insert or update one book and all its tricks.
    Returns number of tricks loaded.
    """
    archive_id = book_data["archive_id"]
    title = book_data["title"]
    pub_year = book_data.get("pub_year")
    publisher = book_data.get("publisher")
    page_count = book_data.get("page_count")
    authors = book_data.get("authors", [])
    subjects = book_data.get("subjects", [])

    # Upsert book
    existing = conn.execute(
        "SELECT id FROM books WHERE archive_id = ?", (archive_id,)
    ).fetchone()

    if existing:
        book_id = existing["id"]
    else:
        cur = conn.execute(
            "INSERT INTO books (archive_id, title, pub_year, publisher, page_count) VALUES (?,?,?,?,?)",
            (archive_id, title, pub_year, publisher, page_count),
        )
        book_id = cur.lastrowid

    # Link authors
    for name in authors:
        if not name:
            continue
        pid = upsert_person(conn, name_cache, name)
        conn.execute(
            "INSERT OR IGNORE INTO book_persons (book_id, person_id, role) VALUES (?,?,'author')",
            (book_id, pid),
        )

    # Link subjects (only if different from authors)
    for name in subjects:
        if not name:
            continue
        pid = upsert_person(conn, name_cache, name)
        conn.execute(
            "INSERT OR IGNORE INTO book_persons (book_id, person_id, role) VALUES (?,?,'subject')",
            (book_id, pid),
        )

    # Tricks
    tricks_loaded = 0
    for trick in book_data.get("tricks", []):
        t_title = trick.get("title", "").strip()
        if not t_title:
            continue

        # Apply junk title filter
        if _is_junk_title(t_title):
            continue

        categories = trick.get("categories", [])
        description = trick.get("description")
        page_number = trick.get("page_number")
        effect_cat = trick.get("effect_category")
        creator_names = trick.get("creators", [])

        # Rule-based prop extraction from breadcrumbs
        props = extract_props(categories)

        # Primary creator → inventor FK
        credited_str = None
        inventor_id = None
        if creator_names:
            credited_str = ", ".join(creator_names)
            inventor_id = upsert_person(conn, name_cache, creator_names[0])
            # Increment trick_count for all credited persons
            for cname in creator_names:
                cpid = upsert_person(conn, name_cache, cname)
                conn.execute(
                    "UPDATE persons SET trick_count = trick_count + 1 WHERE id = ?",
                    (cpid,),
                )

        # Build embed_text
        embed_text = build_embed_text(t_title, description, title, authors, categories)

        conn.execute(
            """INSERT INTO tricks
               (book_id, title, page_number, description, effect_category,
                props, credited_to, inventor_id, embed_text)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                book_id,
                t_title,
                page_number,
                description,
                effect_cat,
                props,
                credited_str,
                inventor_id,
                embed_text,
            ),
        )
        tricks_loaded += 1

    return tricks_loaded


# ── Main ────────────────────────────────────────────────────────────────────────


def run(resume: bool = False, reset: bool = False):
    db_path = paths.db
    raw_dir = paths.raw_data
    checkpoint = Checkpoint("processor", paths.checkpoints)

    if reset:
        logger.info("Reset: wiping DB and checkpoint.")
        if db_path.exists():
            db_path.unlink()
        checkpoint.reset()

    init_schema(db_path)
    conn = get_connection(db_path)

    # Pre-populate name cache from existing persons (for resume)
    name_cache: dict = {}
    for row in conn.execute("SELECT id, name FROM persons"):
        name_cache[row["name"]] = row["id"]

    # Skip .tmp files left by a crashed scraper run
    batch_files = sorted(
        f for f in Path(raw_dir).glob("batch_*.json") if not f.name.endswith(".tmp")
    )
    logger.info(f"Found {len(batch_files)} batch files in {raw_dir}")

    total_books = 0
    total_tricks = 0
    COMMIT_EVERY = 10  # commit every N books within a batch

    for i, bf in enumerate(batch_files):
        batch_key = bf.name
        if checkpoint.is_done(batch_key):
            continue

        try:
            with open(bf, encoding="utf-8") as f:
                books = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Skipping corrupt batch file {bf.name}: {e}")
            checkpoint.mark_done(batch_key)
            continue

        if not isinstance(books, list):
            logger.error(f"Skipping {bf.name}: expected list, got {type(books)}")
            checkpoint.mark_done(batch_key)
            continue

        for j, book_data in enumerate(books):
            try:
                n = load_book(book_data, conn, name_cache)
                total_books += 1
                total_tricks += n
            except Exception as e:
                logger.error(
                    f"Error loading book archive_id="
                    f"{book_data.get(chr(39)+'archive_id'+chr(39), chr(39)+'?'+chr(39))} from {bf.name}: {e}"
                )
            if (j + 1) % COMMIT_EVERY == 0:
                conn.commit()

        conn.commit()
        checkpoint.mark_done(batch_key)

        if (i + 1) % 10 == 0 or i == len(batch_files) - 1:
            logger.info(
                f"Progress: {i+1}/{len(batch_files)} batches | "
                f"{total_books} books | {total_tricks} tricks"
            )

    # Final counts
    b_count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    t_count = conn.execute("SELECT COUNT(*) FROM tricks").fetchone()[0]
    p_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    logger.info(f"Done. DB: {b_count} books | {t_count} tricks | {p_count} persons")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MagiMatch v5 processor")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run(resume=args.resume, reset=args.reset)
