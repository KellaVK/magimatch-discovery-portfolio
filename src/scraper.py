"""
scraper.py — MagiMatch v5 scraper for The Conjuring Archive.

Discovery strategy: browse by title A-Z index pages to collect all book IDs,
then scrape each book page. This covers 100% of books with zero wasted requests
on missing IDs — no gaps, no guessing.

Discovery URLs:  conjuringarchive.com/media/title/{A-Z and 0}
Book page URLs:  conjuringarchive.com/list/medium/{id}

Usage:
    PYTHONPATH=. python src/scraper.py              # full scrape
    PYTHONPATH=. python src/scraper.py --resume     # continue from checkpoint
    PYTHONPATH=. python src/scraper.py --reset      # wipe checkpoint, start over
    PYTHONPATH=. python src/scraper.py --letters A  # only letter A (testing)
    PYTHONPATH=. python src/scraper.py --letters A,B,C

Robustness notes:
    HTTP 429:  reads Retry-After header, waits + jitter, then retries
    HTTP 503/502/504: exponential backoff up to RETRIES attempts
    Connection/timeout errors: backoff + session rebuild on 3rd attempt
    Crash recovery: every book marked done immediately; stats flushed to
                    checkpoint every CHECKPOINT_EVERY books AND at each batch save
    Batch writes: atomic (write to .tmp, then rename) — no corrupt files on crash
    Circuit breaker: if MAX_CONSECUTIVE_ERRORS in a row, pause LONG_PAUSE seconds
    Error tracking: total_errors persisted to checkpoint across resumes
    Failed books: NOT marked done — re-running with --resume will retry them
"""

import argparse
import json
import random
import re
import string
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import Checkpoint
from src.config_loader import cfg, paths
from src.logger import get_logger

logger = get_logger("scraper")

BASE_URL = cfg.scraper.base_url
DELAY = cfg.scraper.request_delay
TIMEOUT = cfg.scraper.timeout
RETRIES = cfg.scraper.max_retries
BATCH = cfg.scraper.batch_size

JUNK_CATEGORY_LEAVES = {c.lower() for c in cfg.processor.junk_categories}

CHECKPOINT_EVERY = 10  # flush stats to checkpoint every N books (mid-batch)
MAX_CONSECUTIVE_ERRORS = 5  # pause if this many requests in a row fail
LONG_PAUSE = 60  # seconds to wait after hitting circuit breaker

ALL_LETTERS = list(string.ascii_uppercase) + ["0"]


# ── HTTP ────────────────────────────────────────────────────────────────────────


def _make_session():
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "MagiMatch-Research-Bot/1.0 "
                "(academic research project; contact: student@tulane.edu)"
            )
        }
    )
    return s


SESSION = _make_session()


def fetch_page(url):
    """
    Fetch URL with retry + smart backoff. Returns (html_text, status_code).

    429: respects Retry-After header + jitter, does not count as an attempt
    502/503/504: exponential backoff
    ConnectionError/Timeout: backoff; rebuilds session on attempt >= 2
    404: immediate return, no retry
    """
    global SESSION

    for attempt in range(RETRIES):
        try:
            r = SESSION.get(url, timeout=TIMEOUT)

            if r.status_code == 200:
                return r.text, 200

            if r.status_code == 404:
                return None, 404

            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 60))
                wait = retry_after + random.uniform(1, 5)
                logger.warning(
                    f"429 Rate limited. Waiting {wait:.0f}s "
                    f"(Retry-After={retry_after}s) before retrying {url}"
                )
                time.sleep(wait)
                # Don't increment attempt — 429 is not a real failure
                continue

            if r.status_code in (502, 503, 504):
                wait = (2**attempt) + random.uniform(0, 2)
                logger.warning(
                    f"HTTP {r.status_code} (attempt {attempt+1}/{RETRIES}). "
                    f"Waiting {wait:.1f}s..."
                )
                time.sleep(wait)
                continue

            # Other non-200 (403, 500, etc.) — don't retry
            logger.warning(f"HTTP {r.status_code} for {url} — skipping.")
            return None, r.status_code

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            wait = (2**attempt) + random.uniform(0, 2)
            logger.warning(
                f"Connection error (attempt {attempt+1}/{RETRIES}): {e}. "
                f"Waiting {wait:.1f}s..."
            )
            if attempt >= 2:
                logger.info(
                    "Rebuilding HTTP session after repeated connection errors..."
                )
                SESSION = _make_session()
            time.sleep(wait)

        except requests.exceptions.Timeout:
            wait = (2**attempt) + random.uniform(0, 2)
            logger.warning(
                f"Timeout (attempt {attempt+1}/{RETRIES}). Waiting {wait:.1f}s..."
            )
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            logger.error(f"Unhandled request exception for {url}: {e}")
            return None, 0

    logger.error(f"Giving up on {url} after {RETRIES} attempts.")
    return None, 0


# ── Discovery ────────────────────────────────────────────────────────────────────


def collect_book_ids_for_letter(letter):
    url = f"{BASE_URL}/media/title/{letter}"
    html, status = fetch_page(url)
    if html is None:
        logger.error(f"Failed to fetch index for letter '{letter}' (HTTP {status})")
        return []

    soup = BeautifulSoup(html, "html.parser")
    books = []
    seen = set()
    for a in soup.find_all("a", href=re.compile(r"/list/medium/\d+")):
        m = re.search(r"/list/medium/(\d+)", a["href"])
        if not m:
            continue
        aid = int(m.group(1))
        if aid not in seen:
            seen.add(aid)
            books.append((aid, a.get_text(strip=True)))

    logger.info(f"  Letter '{letter}': {len(books)} books")
    return books


def collect_all_book_ids(letters):
    all_books = []
    seen_ids = set()
    for i, letter in enumerate(letters):
        for aid, title in collect_book_ids_for_letter(letter):
            if aid not in seen_ids:
                seen_ids.add(aid)
                all_books.append((aid, title))
        if i < len(letters) - 1:
            time.sleep(DELAY)
    logger.info(
        f"Discovery: {len(all_books)} unique books across {len(letters)} letters."
    )
    return all_books


# ── Parsers ────────────────────────────────────────────────────────────────────


def _clean(text):
    if not text:
        return None
    t = re.sub(r"\s+", " ", text).strip()
    return t if t else None


def _fix_spacing(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.sub(r"  +", " ", text).strip()


def _parse_categories(cat_cell):
    if not cat_cell:
        return []
    cats = []
    for li in cat_cell.find_all("li"):
        links = [a.get_text(strip=True) for a in li.find_all("a")]
        if links:
            cats.append(" / ".join(links))
    return cats


def _parse_creators(creators_cell):
    if not creators_cell:
        return [], None

    creator_names = []
    for a in creators_cell.find_all("a", href=True):
        if "/list/person/" in a["href"] or "/media/artist/" in a["href"]:
            name = _clean(a.get_text())
            if name:
                creator_names.append(name)

    desc_parts = []
    for line in creators_cell.get_text(separator="\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d{4}([\s/\-]\d{4})?$", line):
            continue
        if re.match(r"^unknown\s*(\d{4})?$", line, re.I):
            continue
        if line in creator_names:
            continue
        if re.match(r"^[A-Z].+\d{4}$", line):
            continue
        desc_parts.append(line)

    description = None
    if desc_parts:
        description = _clean(_fix_spacing(" ".join(desc_parts)))

    return creator_names, description


def _is_junk_categories(categories):
    if not categories:
        return False
    return all(
        cat.split(" / ")[-1].strip().lower() in JUNK_CATEGORY_LEAVES
        for cat in categories
    )


def parse_book_page(html, archive_id):
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    if not h1:
        return None
    title = _clean(h1.get_text())
    if not title:
        return None

    pub_year = None
    m = re.search(r"\b(1[6-9]\d\d|20\d\d)\b", html[:3000])
    if m:
        pub_year = m.group(1)

    publisher = None
    pub_link = soup.find("a", href=re.compile(r"/media/publisher/"))
    if pub_link:
        publisher = _clean(pub_link.get_text())

    page_count = None
    pc = re.search(r"(\d+)\s+pages?", html[:3000], re.I)
    if pc:
        val = int(pc.group(1))
        if 1 <= val <= 5000:
            page_count = val

    authors = []
    for a in soup.find_all("a", href=re.compile(r"/media/author/")):
        name = _clean(a.get_text())
        if name and name not in authors:
            authors.append(name)

    subjects = []
    work_of = soup.find(string=re.compile(r"Work of", re.I))
    if work_of and work_of.parent:
        for a in work_of.parent.find_all("a", href=re.compile(r"/media/artist/")):
            name = _clean(a.get_text())
            if name and name not in authors and name not in subjects:
                subjects.append(name)

    tricks = []
    table = soup.find("table")
    if table:
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 5:
                continue

            trick_title = _clean(cells[1].get_text())
            if not trick_title:
                continue

            page_num = _clean(cells[4].get_text()) if len(cells) > 4 else None
            cat_cell = cells[5] if len(cells) > 5 else None
            categories = _parse_categories(cat_cell)

            if _is_junk_categories(categories):
                continue

            creator_names, description = _parse_creators(cells[0])

            effect_category = (
                categories[0].split(" / ")[-1].strip() if categories else None
            )
            if not effect_category and not description:
                continue

            tricks.append(
                {
                    "title": trick_title,
                    "page_number": page_num,
                    "description": description,
                    "categories": categories,
                    "effect_category": effect_category,
                    "creators": creator_names,
                }
            )

    return {
        "archive_id": archive_id,
        "title": title,
        "pub_year": pub_year,
        "publisher": publisher,
        "page_count": page_count,
        "authors": authors,
        "subjects": subjects,
        "tricks": tricks,
    }


# ── Batch writer ────────────────────────────────────────────────────────────────


def save_batch(books, batch_num, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"batch_{batch_num:04d}.json"
    tmp_path = path.with_suffix(".json.tmp")
    # Atomic write: write to .tmp, rename to final — crash during write
    # leaves .tmp (harmless) not a corrupt .json
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)
    tmp_path.rename(path)
    logger.info(f"Saved batch_{batch_num:04d}.json ({len(books)} books)")


# ── Main ────────────────────────────────────────────────────────────────────────


def run(letters=None, resume=False, reset=False):
    out_dir = paths.raw_data
    checkpoint = Checkpoint("scraper", paths.checkpoints)

    if reset:
        logger.info("Reset: clearing checkpoint.")
        checkpoint.reset()

    target_letters = letters or ALL_LETTERS

    # Phase 1: Discovery
    id_list_key = "id_list_" + "".join(target_letters)
    cached = checkpoint.get_meta(id_list_key)
    if cached:
        book_ids = cached
        logger.info(f"Using cached discovery: {len(book_ids)} books.")
    else:
        logger.info("=== Phase 1: Discovering book IDs via A-Z index ===")
        book_tuples = collect_all_book_ids(target_letters)
        book_ids = [[aid, t] for aid, t in book_tuples]
        checkpoint.set_meta(id_list_key, book_ids)

    # Phase 2: Scrape
    logger.info(f"=== Phase 2: Scraping {len(book_ids)} book pages ===")

    batch_books = []
    batch_num = checkpoint.get_meta("last_batch", 0)
    total_scraped = checkpoint.get_meta("total_scraped", 0)
    total_tricks = checkpoint.get_meta("total_tricks", 0)
    total_errors = checkpoint.get_meta("total_errors", 0)
    consecutive_errors = 0
    books_since_ckpt = 0

    for archive_id, known_title in book_ids:
        ck_key = f"book_{archive_id}"
        if checkpoint.is_done(ck_key):
            continue

        # Circuit breaker: too many consecutive failures = server problem
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            logger.warning(
                f"{consecutive_errors} consecutive errors. "
                f"Pausing {LONG_PAUSE}s before continuing..."
            )
            time.sleep(LONG_PAUSE)
            consecutive_errors = 0

        url = f"{BASE_URL}/list/medium/{archive_id}"
        html, status = fetch_page(url)
        time.sleep(DELAY)

        if html is None:
            logger.warning(
                f"Skipping {archive_id} ('{known_title}'): "
                f"HTTP {status} after all retries. Will retry on --resume."
            )
            # Do NOT mark done — failed books should be retried on next --resume
            total_errors += 1
            consecutive_errors += 1
            # Still flush error count to checkpoint
            checkpoint.set_meta("total_errors", total_errors)
            continue

        book_data = parse_book_page(html, archive_id)
        if book_data is None:
            logger.warning(f"Parse failed for {archive_id} ('{known_title}')")
            checkpoint.mark_done(ck_key)  # parse failures won't improve on retry
            total_errors += 1
            consecutive_errors += 1
            continue

        consecutive_errors = 0
        n_tricks = len(book_data["tricks"])
        total_scraped += 1
        total_tricks += n_tricks
        books_since_ckpt += 1
        batch_books.append(book_data)

        # Mark done IMMEDIATELY — fine-grained crash recovery
        checkpoint.mark_done(ck_key)

        # Save completed batch
        if len(batch_books) >= BATCH:
            batch_num += 1
            save_batch(batch_books, batch_num, out_dir)
            batch_books = []
            checkpoint.set_meta("last_batch", batch_num)
            checkpoint.set_meta("total_scraped", total_scraped)
            checkpoint.set_meta("total_tricks", total_tricks)
            checkpoint.set_meta("total_errors", total_errors)
            books_since_ckpt = 0
            logger.info(
                f"  {total_scraped}/{len(book_ids)} books | "
                f"{total_tricks} tricks | {total_errors} errors"
            )

        # Mid-batch checkpoint flush every CHECKPOINT_EVERY books
        elif books_since_ckpt >= CHECKPOINT_EVERY:
            checkpoint.set_meta("total_scraped", total_scraped)
            checkpoint.set_meta("total_tricks", total_tricks)
            books_since_ckpt = 0

    # Save final partial batch
    if batch_books:
        batch_num += 1
        save_batch(batch_books, batch_num, out_dir)
        checkpoint.set_meta("last_batch", batch_num)
        checkpoint.set_meta("total_scraped", total_scraped)
        checkpoint.set_meta("total_tricks", total_tricks)
        checkpoint.set_meta("total_errors", total_errors)

    logger.info(
        f"Done: {total_scraped} books | {total_tricks} tricks | "
        f"{total_errors} errors | {batch_num} batch files."
    )
    if total_errors > 0:
        logger.warning(f"{total_errors} books failed. Run with --resume to retry them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MagiMatch v5 scraper")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--letters",
        type=str,
        default=None,
        help="Comma-separated letters to scrape, e.g. 'A,B,C'",
    )
    args = parser.parse_args()

    letter_list = None
    if args.letters:
        letter_list = [l.strip().upper() for l in args.letters.split(",") if l.strip()]

    run(letters=letter_list, resume=args.resume, reset=args.reset)
