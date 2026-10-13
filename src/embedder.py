"""
embedder.py — Step 5: Build rich embed_text and generate .npy embeddings.

What changed from v4:
  - Builds embed_text on-the-fly from ALL useful columns rather than reading
    the stale stored embed_text (which only had title + raw scraped junk).
  - Joins books + persons to pull in effect_category, props, credited_to,
    and style_summary — the fields that actually drive semantic search quality.
  - Updates tricks.embed_text in the DB so the query engine can surface it.
  - Fixes incremental ID tracking bug from v4.
  - Adds ETA logging and per-batch progress.

Usage:
    PYTHONPATH=. python src/embedder.py           # full run
    PYTHONPATH=. python src/embedder.py --resume  # resume after crash
    PYTHONPATH=. python src/embedder.py --reset   # wipe and start over
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import Checkpoint
from src.config_loader import cfg, paths
from src.db import get_connection
from src.logger import get_logger
from src.openai_client import embed_texts

logger = get_logger("embedder")

MODEL = cfg.embeddings.model  # text-embedding-3-small
BATCH_SIZE = cfg.embeddings.batch_size  # 512


# ── Rich embed_text builder ────────────────────────────────────────────────────


def build_embed_text(
    title: str,
    effect_category: str,
    props: str,
    credited_to: str,
    style_summary: str,
    book_title: str,
    pub_year: str,
) -> str:
    """
    Build a keyword-rich string for embedding.

    Field priority (most to least useful for semantic search):
      1. effect_category  — always present, most discriminative
      2. title            — always present
      3. props            — 37% present, very useful for "no gimmicks" style queries
      4. credited_to      — 88% present, useful for person-style queries
      5. style_summary    — 30% present, rich style keywords from the enricher
      6. book / year      — always present, useful for book-level browsing
    """
    parts = []

    # Effect category first — this is the strongest semantic signal
    if effect_category:
        parts.append(f"Effect: {effect_category}")

    # Trick title
    parts.append(title)

    # Props — normalize JSON blobs to clean text before embedding
    # e.g. {"props": ["playing cards"], "gimmick": false} → "playing cards, no gimmick"
    if props:
        try:
            parsed = json.loads(props)
            prop_list = parsed.get("props", [])
            gimmick = parsed.get("gimmick", None)
            prop_str = ", ".join(prop_list) if prop_list else ""
            if gimmick is True:
                prop_str = (prop_str + ", gimmick").strip(", ")
            elif gimmick is False:
                prop_str = (prop_str + ", no gimmick").strip(", ")
            if prop_str:
                parts.append(f"Props: {prop_str}")
        except (json.JSONDecodeError, TypeError):
            # Already plain text (e.g. "cards" or "Egg, Spongeball, balls")
            parts.append(f"Props: {props}")

    # Creator
    if credited_to:
        parts.append(f"By: {credited_to}")

    # Style summary from the enricher — rich keyword text describing the creator's style
    if style_summary:
        parts.append(f"Style: {style_summary.strip()}")

    # Book context
    book_part = book_title or ""
    if pub_year:
        book_part += f" ({pub_year})"
    if book_part:
        parts.append(f"From: {book_part}")

    return ". ".join(parts)


# ── Load all tricks with joined context ───────────────────────────────────────


def load_tricks(conn) -> list:
    """
    Fetch all tricks joined with book info and the primary inventor's style_summary.
    Returns list of dicts with everything needed for build_embed_text.
    """
    rows = conn.execute("""
        SELECT
            t.id,
            t.title,
            t.effect_category,
            t.props,
            t.credited_to,
            b.title       AS book_title,
            b.pub_year,
            p.style_summary
        FROM tricks t
        JOIN books b ON b.id = t.book_id
        LEFT JOIN persons p ON p.id = t.inventor_id
        ORDER BY t.id
        """).fetchall()
    return [dict(r) for r in rows]


# ── Main run function ──────────────────────────────────────────────────────────


def run(resume: bool = False, reset: bool = False):
    emb_dir = paths.embeddings
    emb_dir.mkdir(parents=True, exist_ok=True)
    tricks_npy = emb_dir / "tricks.npy"
    ids_npy = emb_dir / "trick_ids.npy"
    checkpoint = Checkpoint("embedder", paths.checkpoints)

    if reset:
        for f in [tricks_npy, ids_npy]:
            if f.exists():
                f.unlink()
        checkpoint.reset()
        logger.info("Reset complete.")

    conn = get_connection(paths.db)
    tricks = load_tricks(conn)
    logger.info(f"Loaded {len(tricks):,} tricks from DB.")

    # Build embed_text for every trick
    texts = []
    for t in tricks:
        texts.append(
            build_embed_text(
                title=t["title"] or "",
                effect_category=t["effect_category"] or "",
                props=t["props"] or "",
                credited_to=t["credited_to"] or "",
                style_summary=t["style_summary"] or "",
                book_title=t["book_title"] or "",
                pub_year=t["pub_year"] or "",
            )
        )

    trick_ids = [t["id"] for t in tricks]

    # Log a few samples so you can inspect quality
    logger.info("Sample embed_texts:")
    for i in [0, 100, 1000, 5000]:
        if i < len(texts):
            logger.info(f"  [{i}] {texts[i][:120]}")

    # Resume: load existing partial embeddings
    start = 0
    all_embs = []
    if resume and tricks_npy.exists() and ids_npy.exists():
        all_embs = list(np.load(str(tricks_npy), allow_pickle=False))
        start = len(all_embs)
        logger.info(f"Resuming from {start:,} already embedded.")

    remaining_texts = texts[start:]
    total_batches = (len(remaining_texts) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        f"Embedding {len(remaining_texts):,} tricks in {total_batches} batches "
        f"(batch_size={BATCH_SIZE}, model={MODEL})..."
    )

    SAVE_EVERY = 10  # write to disk every N batches, not every batch

    t0 = time.time()
    for i in range(0, len(remaining_texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = remaining_texts[i : i + BATCH_SIZE]

        embs = embed_texts(batch, model=MODEL, batch_size=BATCH_SIZE)
        all_embs.extend(embs)

        # ETA
        elapsed = time.time() - t0
        per_batch = elapsed / batch_num
        eta = (total_batches - batch_num) * per_batch
        logger.info(
            f"  Batch {batch_num}/{total_batches} | "
            f"{len(all_embs):,} embedded | "
            f"ETA {eta/60:.1f} min"
        )

        # Save every SAVE_EVERY batches (not every batch — avoid growing write cost)
        if batch_num % SAVE_EVERY == 0:
            saved_count = start + len(all_embs)
            np.save(str(tricks_npy), np.array(all_embs, dtype=np.float32))
            np.save(str(ids_npy), np.array(trick_ids[:saved_count], dtype=np.int64))
            logger.info(f"  Checkpoint saved at batch {batch_num}.")

    # Final save
    final_embs = np.array(all_embs, dtype=np.float32)
    final_ids = np.array(trick_ids, dtype=np.int64)
    np.save(str(tricks_npy), final_embs)
    np.save(str(ids_npy), final_ids)
    logger.info(f"Embeddings saved: {final_embs.shape} → {tricks_npy}")

    # Update embed_text column in DB so the query engine can surface it
    logger.info("Updating embed_text in DB...")
    updates = [(texts[i], trick_ids[i]) for i in range(len(tricks))]
    conn.executemany("UPDATE tricks SET embed_text = ? WHERE id = ?", updates)
    conn.commit()
    logger.info("embed_text column updated.")

    # Log to embedding_log table
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_log "
            "(model TEXT, record_count INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO embedding_log (model, record_count) VALUES (?,?)",
            (MODEL, len(all_embs)),
        )
        conn.commit()
    except Exception:
        pass

    conn.close()
    total_time = (time.time() - t0) / 60
    logger.info(f"Done in {total_time:.1f} min. {len(all_embs):,} embeddings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate MagiMatch embeddings")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last saved batch"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Wipe existing embeddings and start over"
    )
    args = parser.parse_args()
    run(resume=args.resume, reset=args.reset)
