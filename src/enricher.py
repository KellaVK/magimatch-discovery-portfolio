"""
magimatch/src/enricher.py — MagiMatch enrichment script (v2)

Three fully independent passes:

  --persons     Fill persons.style_summary via GPT
  --categories  Fill tricks.effect_category (rule pre-pass → batched GPT)
  --props       Fill tricks.props (rule pre-pass → batched GPT)

All passes are idempotent — re-running only touches NULL rows.
Batched GPT calls (20 tricks per request) keep cost and time low.

Usage:
    PYTHONPATH=. python src/enricher.py --categories
    PYTHONPATH=. python src/enricher.py --props
    PYTHONPATH=. python src/enricher.py --persons
    PYTHONPATH=. python src/enricher.py --persons --categories
    PYTHONPATH=. python src/enricher.py --dry-run --categories
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import cfg, paths
from src.db import get_connection
from src.logger import get_logger
from src.openai_client import chat_completion

logger = get_logger("enricher")

MIN_TRICKS = cfg.enrichment.min_tricks_for_style
BATCH_SIZE = 20  # tricks per GPT call
COMMIT_EVERY = 200  # rows between DB commits
RATE_DELAY = 0.15  # seconds between API calls


# ── TEXT CLEANING ─────────────────────────────────────────────────────────────

# Words that appear in descriptions but are noise for classification
_NOISE_WORDS = {
    "unknown",
    "with",
    "from",
    "into",
    "using",
    "then",
    "method",
    "routine",
    "trick",
    "performer",
    "variation",
    "handling",
    "very",
    "brief",
    "also",
    "chapter",
    "intro",
    "section",
    "part",
    "notes",
    "volume",
    "lecture",
    "translation",
    "described",
    "effect",
    "version",
    "another",
    "original",
    "reprinted",
    "discussion",
    "explanation",
    "alternate",
    "improved",
    "revised",
    "similar",
    "different",
    "same",
    "basic",
    "simple",
    "first",
    "second",
    "third",
    "number",
    "page",
    "book",
    "show",
    "each",
    "both",
    "their",
    "that",
    "this",
    "which",
    "when",
    "where",
    "what",
    "have",
    "been",
    "will",
    "more",
    "than",
    "only",
    "just",
    "used",
    "work",
    "made",
}


def _clean_description(title: str, desc: str | None) -> str:
    """Strip noise from description: repeated title text, newlines, author names."""
    if not desc:
        return ""
    # Collapse \n sequences into spaces
    text = re.sub(r"\\n|\n", " ", desc)
    # Remove the trick title repeated at the start (common artifact)
    title_lower = title.lower().strip()
    text_lower = text.lower().strip()
    if text_lower.startswith(title_lower):
        text = text[len(title) :].strip()
    # Remove stray asterisks and leading punctuation
    text = re.sub(r"\s*\*+\s*", " ", text)
    text = text.strip(" \\n,.-")
    return text.strip()


def _search_text(title: str, desc: str | None, category: str | None = None) -> str:
    """Combined lowercase search string for rule matching."""
    parts = [title or "", _clean_description(title, desc), category or ""]
    return " ".join(parts).lower()


# ── PASS 1: PERSONS — style_summary ──────────────────────────────────────────

_STYLE_SYSTEM = (
    "You describe the distinctive magical style of a magician or magic author. "
    "Your description is used as a semantic search index — it must be keyword-rich and specific. "
    "Focus on: what props they use, what techniques they're known for, what makes their approach unique. "
    "Do NOT just say 'card magician' — be specific about what kind of card work, what their philosophy is. "
    "Respond with 30-50 words only. No preamble."
)


def run_persons(conn, dry_run: bool = False):
    rows = conn.execute(
        """
        SELECT p.id, p.name,
               COUNT(t.id) AS trick_count
        FROM persons p
        JOIN tricks t ON t.inventor_id = p.id
        WHERE p.style_summary IS NULL
        GROUP BY p.id
        HAVING trick_count >= ?
        ORDER BY trick_count DESC
        """,
        (MIN_TRICKS,),
    ).fetchall()

    logger.info(
        f"[persons] {len(rows)} persons to enrich (trick_count >= {MIN_TRICKS})"
    )
    if not rows:
        logger.info("[persons] Nothing to do.")
        return

    if dry_run:
        for r in rows[:10]:
            logger.info(f"  would enrich: {r['name']} ({r['trick_count']} tricks)")
        logger.info("  (dry run — no changes written)")
        return

    enriched = 0
    for row in rows:
        pid = row["id"]
        name = row["name"]

        tricks = conn.execute(
            """
            SELECT title, description, effect_category
            FROM tricks WHERE inventor_id = ?
            ORDER BY RANDOM() LIMIT 20
            """,
            (pid,),
        ).fetchall()

        lines = []
        for t in tricks:
            line = t["title"]
            if t["effect_category"]:
                line += f" [{t['effect_category']}]"
            cleaned = _clean_description(t["title"], t["description"])
            if cleaned:
                line += f" — {cleaned}"
            lines.append(line)

        sample = "\n".join(f"- {l}" for l in lines)
        user_prompt = (
            f"Magician/author: {name}\n\n"
            f"Sample of their credited tricks:\n{sample}\n\n"
            f"Describe their distinctive magical style in 30-50 words."
        )

        try:
            summary = chat_completion(
                system=_STYLE_SYSTEM,
                prompt=user_prompt,
                max_tokens=120,
                temperature=0.3,
            ).strip()

            conn.execute(
                "UPDATE persons SET style_summary = ? WHERE id = ?",
                (summary, pid),
            )
            enriched += 1

            if enriched % COMMIT_EVERY == 0:
                conn.commit()
                logger.info(f"  [persons] checkpoint: {enriched}/{len(rows)}")

            logger.info(f"  ✓ {name}: {summary[:80]}…")

        except Exception as e:
            logger.warning(f"  ✗ {name}: {e}")

        time.sleep(RATE_DELAY)

    conn.commit()
    logger.info(f"[persons] done — {enriched}/{len(rows)} enriched.")


# ── PASS 2: CATEGORIES — effect_category ─────────────────────────────────────

VALID_CATEGORIES = [
    "Single Card",
    "Multiple Cards",
    "Packet Tricks",
    "Ambitious Card",
    "Card to Impossible Location",
    "Gambling Demonstrations",
    "False Deals",
    "Memory Systems",
    "Memorized Deck",
    "Coin Magic",
    "Bills & Paper Money",
    "Cups & Balls",
    "Thimbles",
    "Rope Magic",
    "Ring Magic",
    "Silk Magic",
    "Mentalism",
    "Book Tests",
    "Predictions",
    "Mind Reading",
    "Psychokinesis",
    "Hypnosis",
    "Bizarre Magic",
    "Stand-up / Platform",
    "Stage Illusions",
    "Children's Magic",
    "Comedy Magic",
    "Escapology",
    "Sponge Balls",
    "Balls & Billiard Balls",
    "Dice",
    "Matches & Cigarettes",
    "Rings & Linking Rings",
    "Banknotes",
    "Mathematical Magic",
    "Puzzle Effects",
    "Flourishes",
    "Theory & Essays",
    "Routines",
    "Vanish",
    "Production",
    "Transposition",
    "Transformation",
    "Restoration",
    "Penetration",
    "Levitation",
    "Special Powers",
    "Miscellaneous Close-up",
]

# Rule-based category signals: (search_terms, category)
# Order matters — more specific rules first
_CATEGORY_RULES: list[tuple[list[str], str]] = [
    # Gambling / false deals (before generic card)
    (
        [
            "second deal",
            "bottom deal",
            "center deal",
            "false deal",
            "gambling demo",
            "false shuffle",
            "zarrow",
            "faro shuffle",
            "riffle stack",
        ],
        "Gambling Demonstrations",
    ),
    (["false deal", "base deal", "strike second", "push-off second"], "False Deals"),
    # Memorized deck (before generic card)
    (
        [
            "memorized deck",
            "mnemonica",
            "aronson stack",
            "stack order",
            "stacked deck",
            "si stebbins",
            "eight kings",
        ],
        "Memorized Deck",
    ),
    # Specific card effects
    (["ambitious card"], "Ambitious Card"),
    (
        [
            "card to wallet",
            "card to pocket",
            "card to impossible",
            "card in bottle",
            "card through window",
            "card under glass",
            "card to ceiling",
            "card to box",
        ],
        "Card to Impossible Location",
    ),
    (["packet", "four card", "four-card", "five card", "five-card"], "Packet Tricks"),
    # Generic card — must come AFTER more specific card rules
    (
        [
            "card",
            "cards",
            "deck",
            "aces",
            "selection",
            "chosen card",
            "top card",
            "bottom card",
            "face card",
            "joker",
        ],
        "Single Card",
    ),
    # Coin
    (
        [
            "coin",
            "coins",
            "silver",
            "copper",
            "half dollar",
            "quarter",
            "palming coin",
            "coin vanish",
            "coin production",
            "coin through",
            "coin across",
            "retention vanish",
            "muscle pass",
            "french drop",
        ],
        "Coin Magic",
    ),
    # Bills / money
    (
        [
            "bill",
            "bills",
            "dollar",
            "banknote",
            "money",
            "currency",
            "monopoly",
            "play money",
        ],
        "Bills & Paper Money",
    ),
    # Rope
    (
        ["rope", "ropes", "cut and restored rope", "knot", "knots", "cord", "string"],
        "Rope Magic",
    ),
    # Rings
    (
        ["linking ring", "chinese ring", "ring links", "ring routine"],
        "Rings & Linking Rings",
    ),
    (["ring on", "ring off", "finger ring", "borrowed ring"], "Ring Magic"),
    # Silk
    (["silk", "silks", "handkerchief", "hanky", "foulard"], "Silk Magic"),
    # Cups and balls
    (["cups and balls", "cup and ball", "chop cup"], "Cups & Balls"),
    # Sponge
    (["sponge ball", "sponge balls", "sponge rabbit"], "Sponge Balls"),
    # Thimbles
    (["thimble", "thimbles"], "Thimbles"),
    # Balls
    (
        [
            "billiard ball",
            "billiard balls",
            "multiplying balls",
            "ball production",
            "ball vanish",
        ],
        "Balls & Billiard Balls",
    ),
    # Mentalism / mind reading
    (
        [
            "mind reading",
            "reads mind",
            "divines",
            "thought of",
            "think of",
            "mental",
            "esp",
            "extrasensory",
            "sixth sense",
            "psychic",
            "clairvoyance",
            "telepathy",
            "thought transmission",
        ],
        "Mind Reading",
    ),
    (["book test", "book tests", "any book", "dictionary test"], "Book Tests"),
    (
        [
            "prediction",
            "predicted",
            "predicts",
            "envelope prediction",
            "sealed envelope",
            "headline",
        ],
        "Predictions",
    ),
    (
        [
            "psychokinesis",
            "telekinesis",
            "pk touch",
            "bending",
            "metal bending",
            "spoon bend",
        ],
        "Psychokinesis",
    ),
    (["hypnos", "hypnotism", "trance", "suggestion", "pseudo hypnosis"], "Hypnosis"),
    (
        [
            "mentalism",
            "mentalist",
            "billet",
            "center tear",
            "peek",
            "nail writer",
            "swami",
            "one ahead",
        ],
        "Mentalism",
    ),
    # Stage / platform
    (
        [
            "stage illusion",
            "cabinet",
            "levitat",
            "sawing",
            "sub trunk",
            "substitution trunk",
        ],
        "Stage Illusions",
    ),
    (["stand-up", "stand up", "platform", "parlor", "parlour"], "Stand-up / Platform"),
    # Escape
    (
        [
            "escape",
            "escapology",
            "handcuffs",
            "strait jacket",
            "straitjacket",
            "locked",
        ],
        "Escapology",
    ),
    # Children's
    (
        [
            "children",
            "kids",
            "birthday party",
            "balloon animal",
            "balloon",
            "school show",
        ],
        "Children's Magic",
    ),
    # Comedy
    (
        ["comedy", "comedic", "gag", "joke effect", "sight gag", "humorous"],
        "Comedy Magic",
    ),
    # Bizarre
    (
        [
            "bizarre",
            "gothic",
            "séance",
            "spirit",
            "ghost",
            "haunted",
            "occult",
            "voodoo",
        ],
        "Bizarre Magic",
    ),
    # Matches / cigarettes
    (
        [
            "match",
            "matches",
            "matchbox",
            "cigarette",
            "cigar",
            "smoke",
            "ashtray",
            "lighter",
        ],
        "Matches & Cigarettes",
    ),
    # Dice
    (["dice", "die", "die box"], "Dice"),
    # Mathematical / puzzle
    (
        [
            "calendar",
            "clock",
            "watch",
            "number",
            "mathematical",
            "calculation",
            "puzzle",
            "brainteaser",
            "glass puzzle",
            "coin puzzle",
            "match puzzle",
        ],
        "Mathematical Magic",
    ),
    # Memory
    (
        [
            "memory system",
            "memorize",
            "memorisation",
            "peg system",
            "journey method",
            "names and faces",
        ],
        "Memory Systems",
    ),
    # Flourish
    (
        [
            "flourish",
            "cut",
            "spring",
            "riffle",
            "waterfall",
            "card fan",
            "one-handed cut",
            "charlier",
        ],
        "Flourishes",
    ),
    # Theory
    (
        [
            "theory",
            "essay",
            "philosophy",
            "history of",
            "biography",
            "psychology of",
            "performance theory",
            "misdirection theory",
            "how to perform",
            "presentation theory",
        ],
        "Theory & Essays",
    ),
    # Vanish / production / transposition
    (["vanish", "vanishes", "disappear", "invisible"], "Vanish"),
    (["production", "produces", "appear", "appears", "appear from"], "Production"),
    (
        [
            "transposition",
            "transpose",
            "travel",
            "travels",
            "exchange",
            "switch places",
        ],
        "Transposition",
    ),
    (["transformation", "change", "changes into", "transforms"], "Transformation"),
    (
        ["restoration", "restored", "torn and restored", "cut and restored"],
        "Restoration",
    ),
    (["penetration", "through", "penetrates", "passes through"], "Penetration"),
    (["levitation", "levitate", "float", "floats", "rises"], "Levitation"),
]

# Book-level author → likely category (used when trick has minimal description)
_AUTHOR_CATEGORY_HINTS: dict[str, str] = {
    "david roth": "Coin Magic",
    "david": "Coin Magic",  # matched only when in coin books
    "slydini": "Coin Magic",
    "han ping chien": "Coin Magic",
    "matt mello": "Mentalism",
    "max maven": "Mentalism",
    "luke jermay": "Mentalism",
    "bob cassidy": "Mentalism",
    "annemann": "Mentalism",
    "corinda": "Mentalism",
    "juan tamariz": "Memorized Deck",
    "simon aronson": "Memorized Deck",
    "edward marlo": "Single Card",
    "ed marlo": "Single Card",
    "karl fulves": "Single Card",
    "darwin ortiz": "Single Card",
    "roberto giobbi": "Single Card",
}


def _rule_category(
    title: str, desc: str | None, book_title: str, authors: str | None
) -> str | None:
    """Return a category from rules, or None if ambiguous."""
    text = _search_text(title, desc)
    book_text = (book_title or "").lower()
    authors_lower = (authors or "").lower()

    for signals, category in _CATEGORY_RULES:
        if any(sig in text for sig in signals):
            return category

    # Author hint as fallback for very sparse descriptions
    for author_key, category in _AUTHOR_CATEGORY_HINTS.items():
        if author_key in authors_lower:
            # Only apply hint if book context agrees
            if category == "Coin Magic" and "coin" in book_text:
                return category
            if category == "Mentalism" and any(
                w in book_text for w in ["mental", "mind", "psychic"]
            ):
                return category
            if category in ("Single Card", "Memorized Deck") and any(
                w in book_text
                for w in ["card", "deck", "poker", "marlo", "tamariz", "aronson"]
            ):
                return category

    return None


_CATEGORY_SYSTEM = (
    "You categorize magic tricks for a searchable database. "
    "Given a batch of tricks (title, description, book, authors), assign each the SINGLE best category. "
    "Return a JSON array of category strings in the SAME ORDER as the input. "
    "Choose only from the provided category list. "
    "If nothing fits, use 'Miscellaneous Close-up'. "
    "Return only the JSON array — no markdown, no explanation."
)


def _batch_categorize(batch: list[dict], categories: list[str]) -> list[str]:
    """Send a batch of tricks to GPT and return a list of categories."""
    cat_list = ", ".join(f'"{c}"' for c in categories)
    items = "\n".join(
        f'{i+1}. Title: {r["title"]} | Desc: {_clean_description(r["title"], r["description"])[:120]} '
        f'| Book: {r["book_title"]} | Authors: {r["authors"] or "unknown"}'
        for i, r in enumerate(batch)
    )
    user_prompt = (
        f"Categories: [{cat_list}]\n\n"
        f"Tricks:\n{items}\n\n"
        f"Return a JSON array of {len(batch)} category strings."
    )
    raw = chat_completion(
        system=_CATEGORY_SYSTEM,
        prompt=user_prompt,
        max_tokens=len(batch) * 12,
        temperature=0.0,
    ).strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    parsed = json.loads(raw)

    # Snap each to valid category
    result = []
    for val in parsed:
        cat = val.strip().strip('"')
        result.append(cat if cat in VALID_CATEGORIES else "Miscellaneous Close-up")

    # Pad if GPT returned fewer than expected
    while len(result) < len(batch):
        result.append("Miscellaneous Close-up")

    return result[: len(batch)]


def run_categories(conn, dry_run: bool = False):
    rows = conn.execute("""
        SELECT t.id, t.title, t.description,
               b.title AS book_title,
               GROUP_CONCAT(DISTINCT p.name) AS authors
        FROM tricks t
        JOIN books b ON t.book_id = b.id
        LEFT JOIN book_persons bp ON bp.book_id = b.id AND bp.role = 'author'
        LEFT JOIN persons p ON p.id = bp.person_id
        WHERE t.effect_category IS NULL
        GROUP BY t.id
        """).fetchall()

    logger.info(f"[categories] {len(rows)} tricks missing effect_category")
    if not rows:
        logger.info("[categories] Nothing to do.")
        return

    if dry_run:
        rule_count = sum(
            1
            for r in rows
            if _rule_category(
                r["title"], r["description"], r["book_title"], r["authors"]
            )
        )
        logger.info(f"  rule pre-pass would fill: {rule_count}/{len(rows)}")
        logger.info(
            f"  GPT batches needed: {((len(rows) - rule_count) + BATCH_SIZE - 1) // BATCH_SIZE}"
        )
        logger.info("  (dry run — no changes written)")
        return

    # Pass 1: rule-based
    rule_filled = 0
    gpt_needed = []

    for row in rows:
        cat = _rule_category(
            row["title"], row["description"], row["book_title"], row["authors"]
        )
        if cat:
            conn.execute(
                "UPDATE tricks SET effect_category = ? WHERE id = ?",
                (cat, row["id"]),
            )
            rule_filled += 1
        else:
            gpt_needed.append(dict(row))

    conn.commit()
    logger.info(
        f"  [categories] rule pre-pass: {rule_filled} filled, {len(gpt_needed)} need GPT"
    )

    # Pass 2: batched GPT
    gpt_filled = 0
    errors = 0

    for batch_start in range(0, len(gpt_needed), BATCH_SIZE):
        batch = gpt_needed[batch_start : batch_start + BATCH_SIZE]
        try:
            categories = _batch_categorize(batch, VALID_CATEGORIES)
            for row, cat in zip(batch, categories):
                conn.execute(
                    "UPDATE tricks SET effect_category = ? WHERE id = ?",
                    (cat, row["id"]),
                )
            gpt_filled += len(batch)

            if gpt_filled % COMMIT_EVERY == 0:
                conn.commit()
                total_done = rule_filled + gpt_filled
                logger.info(f"  [categories] checkpoint: {total_done}/{len(rows)}")

        except Exception as e:
            errors += len(batch)
            logger.warning(f"  ✗ batch at {batch_start}: {e}")

        time.sleep(RATE_DELAY)

    conn.commit()
    logger.info(
        f"[categories] done — {rule_filled} rule + {gpt_filled} GPT filled, {errors} errors."
    )


# ── PASS 3: PROPS — props column ─────────────────────────────────────────────

# Explicit prop vocabulary — covers magic and non-magic props
_PROP_VOCAB = """
Playing cards, deck of cards, coin, coins, half dollar, quarter, poker chip,
rope, cord, string, thread, silk, handkerchief, ring, rings, linking rings,
finger ring, borrowed ring, cups, balls, sponge balls, thimble, dice, die,
matchbox, matches, cigarette, cigar, lighter, ashtray, smoke,
dollar bill, banknote, money, bill, paper money, monopoly bill,
envelope, sealed envelope, business card, blank card, index card,
rubber band, rubber bands, paper clip, safety pin, nail, pencil, pen, marker,
newspaper, paper bag, paper, napkin, tissue,
bottle, glass, glasses, cup, mug, vase,
watch, clock, calendar, book, dictionary, magazine,
key, keys, padlock, lock, chain,
sugar cube, salt shaker, pepper shaker, sugar packet,
balloon, balloon animal,
candle, flame, fire,
mirror, photograph, picture frame,
chair, table, box, bag, briefcase, wallet, purse, pocket,
steel ball, marble, golf ball, tennis ball, ball,
handcuffs, rope restraint, strait jacket,
wand, magic wand,
thumb tip, gimmick, special prop
"""

_PROPS_SYSTEM = (
    "You extract physical prop requirements from magic trick descriptions. "
    "Return a JSON array of objects, one per trick, in the SAME ORDER as input. "
    'Each object: {"props": [list of props], "gimmick": true/false}\n'
    "props: physical objects needed. Use specific names from this vocabulary where applicable:\n"
    f"{_PROP_VOCAB}\n"
    "Also include non-standard objects if clearly implied (e.g. 'olive', 'horse', 'tube'). "
    "Empty list [] if truly propless (pure mental effect, no objects). "
    "gimmick: true only if a specially prepared or secret prop is implied. "
    "Return only the JSON array — no markdown, no explanation."
)

# Rule-based prop signals: (search_terms, props_list, gimmick)
_PROP_RULES: list[tuple[list[str], list[str], bool]] = [
    (["cigarette", "cigar"], ["cigarette"], False),
    (["match", "matchbox", "matches"], ["matches", "matchbox"], False),
    (["rope", "cord", "string"], ["rope"], False),
    (["silk", "handkerchief", "hanky", "foulard"], ["silk"], False),
    (["sponge ball", "sponge rabbit"], ["sponge balls"], False),
    (["thimble"], ["thimble"], False),
    (["balloon"], ["balloon"], False),
    (["rubber band"], ["rubber band"], False),
    (["dice", " die "], ["dice"], False),
    (["linking ring", "chinese ring"], ["linking rings"], False),
    (["cups and balls", "chop cup"], ["cups", "balls"], False),
    (["newspaper"], ["newspaper"], False),
    (["candle"], ["candle"], False),
    (
        [
            "dollar bill",
            "banknote",
            "bank note",
            "bill changes",
            "bill vanish",
            "bill appears",
            "bill in",
            "bill through",
            "money",
        ],
        ["dollar bill"],
        False,
    ),
    (["envelope"], ["envelope"], False),
    (["rubber band"], ["rubber band"], False),
    (["wallet"], ["wallet"], False),
    (["watch", "wristwatch"], ["watch"], False),
    (["glass", "glasses"], ["glass"], False),
    (["bottle"], ["bottle"], False),
    (["book test", "any book", "dictionary"], ["book"], False),
    (["nail writer", "swami", "boon writer"], ["nail writer"], True),
    (["marked deck", "marked card", "marked cards"], ["marked deck"], True),
    (
        [
            "invisible deck",
            "svengali",
            "stripper deck",
            "roughed",
            "gaffed deck",
            "gaff deck",
            "double backer",
            "double facer",
            "misprinted",
            "prepared deck",
        ],
        ["gimmicked deck"],
        True,
    ),
]


def _rule_props(title: str, desc: str | None, category: str | None) -> dict | None:
    """Return props dict from rules, or None if ambiguous."""
    text = _search_text(title, desc, category)

    # Pure card tricks with no special props
    if category in (
        "Single Card",
        "Multiple Cards",
        "Packet Tricks",
        "Ambitious Card",
        "Gambling Demonstrations",
        "False Deals",
        "Memorized Deck",
        "Flourishes",
    ) and not any(
        sig in text
        for sig in [
            "coin",
            "rope",
            "silk",
            "bill",
            "envelope",
            "match",
            "cigarette",
            "ring",
            "glass",
            "bottle",
        ]
    ):
        gimmick = any(
            g in text
            for g in [
                "gimmick",
                "gaff",
                "marked",
                "stripper",
                "svengali",
                "roughed",
                "double backer",
                "prepared",
                "invisible deck",
            ]
        )
        return {"props": ["playing cards"], "gimmick": gimmick}

    # Propless mentalism
    if category in (
        "Mentalism",
        "Mind Reading",
        "Predictions",
        "Theory & Essays",
        "Memory Systems",
    ) and not any(
        sig in text
        for sig in [
            "card",
            "coin",
            "rope",
            "bill",
            "envelope",
            "dice",
            "ring",
            "glass",
            "object",
        ]
    ):
        return {"props": [], "gimmick": False}

    for signals, props, gimmick in _PROP_RULES:
        if any(sig in text for sig in signals):
            gimmick_override = gimmick or any(
                g in text for g in ["gimmick", "gaff", "prepared", "special"]
            )
            return {"props": props, "gimmick": gimmick_override}

    return None


def _batch_props(batch: list[dict]) -> list[dict]:
    """Send a batch of tricks to GPT and return a list of props dicts."""
    items = "\n".join(
        f'{i+1}. Title: {r["title"]} | Desc: {_clean_description(r["title"], r["description"])[:120]} '
        f'| Category: {r["effect_category"] or "unknown"}'
        for i, r in enumerate(batch)
    )
    user_prompt = (
        f"Tricks:\n{items}\n\n"
        f"Return a JSON array of {len(batch)} objects: "
        '{"props": [...], "gimmick": true/false}'
    )
    raw = chat_completion(
        system=_PROPS_SYSTEM,
        prompt=user_prompt,
        max_tokens=len(batch) * 30,
        temperature=0.0,
    ).strip()

    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    parsed = json.loads(raw)

    result = []
    for item in parsed:
        if not isinstance(item, dict):
            item = {}
        props = item.get("props", [])
        gimmick = bool(item.get("gimmick", False))
        if not isinstance(props, list):
            props = []
        result.append({"props": props, "gimmick": gimmick})

    # Pad if GPT returned fewer than expected
    while len(result) < len(batch):
        result.append({"props": [], "gimmick": False})

    return result[: len(batch)]


def run_props(conn, dry_run: bool = False):
    rows = conn.execute("""
        SELECT id, title, description, effect_category
        FROM tricks
        WHERE props IS NULL
        """).fetchall()

    logger.info(f"[props] {len(rows)} tricks missing props")
    if not rows:
        logger.info("[props] Nothing to do.")
        return

    if dry_run:
        rule_count = sum(
            1
            for r in rows
            if _rule_props(r["title"], r["description"], r["effect_category"])
        )
        logger.info(f"  rule pre-pass would fill: {rule_count}/{len(rows)}")
        logger.info(
            f"  GPT batches needed: {((len(rows) - rule_count) + BATCH_SIZE - 1) // BATCH_SIZE}"
        )
        logger.info("  (dry run — no changes written)")
        return

    # Pass 1: rule-based
    rule_filled = 0
    gpt_needed = []

    for row in rows:
        result = _rule_props(row["title"], row["description"], row["effect_category"])
        if result:
            conn.execute(
                "UPDATE tricks SET props = ? WHERE id = ?",
                (json.dumps(result), row["id"]),
            )
            rule_filled += 1
        else:
            gpt_needed.append(dict(row))

    conn.commit()
    logger.info(
        f"  [props] rule pre-pass: {rule_filled} filled, {len(gpt_needed)} need GPT"
    )

    # Pass 2: batched GPT
    gpt_filled = 0
    errors = 0

    for batch_start in range(0, len(gpt_needed), BATCH_SIZE):
        batch = gpt_needed[batch_start : batch_start + BATCH_SIZE]
        try:
            results = _batch_props(batch)
            for row, result in zip(batch, results):
                conn.execute(
                    "UPDATE tricks SET props = ? WHERE id = ?",
                    (json.dumps(result), row["id"]),
                )
            gpt_filled += len(batch)

            if gpt_filled % COMMIT_EVERY == 0:
                conn.commit()
                total_done = rule_filled + gpt_filled
                logger.info(f"  [props] checkpoint: {total_done}/{len(rows)}")

        except Exception as e:
            errors += len(batch)
            logger.warning(f"  ✗ batch at {batch_start}: {e}")

        time.sleep(RATE_DELAY)

    conn.commit()
    logger.info(
        f"[props] done — {rule_filled} rule + {gpt_filled} GPT filled, {errors} errors."
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MagiMatch enrichment — run any combination of passes independently.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  PYTHONPATH=. python src/enricher.py --categories
  PYTHONPATH=. python src/enricher.py --props
  PYTHONPATH=. python src/enricher.py --persons
  PYTHONPATH=. python src/enricher.py --persons --categories
  PYTHONPATH=. python src/enricher.py --dry-run --categories
        """,
    )
    parser.add_argument(
        "--persons", action="store_true", help="Fill persons.style_summary"
    )
    parser.add_argument(
        "--categories", action="store_true", help="Fill tricks.effect_category"
    )
    parser.add_argument("--props", action="store_true", help="Fill tricks.props")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview only — no DB writes"
    )
    args = parser.parse_args()

    if not any([args.persons, args.categories, args.props]):
        parser.print_help()
        print("\nError: specify at least one pass (--persons, --categories, --props)")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN — no changes will be written to the database.")

    conn = get_connection(paths.db)

    if args.persons:
        logger.info("=" * 55)
        logger.info("PASS: persons (style_summary)")
        logger.info("=" * 55)
        run_persons(conn, dry_run=args.dry_run)

    if args.categories:
        logger.info("=" * 55)
        logger.info("PASS: categories (effect_category)")
        logger.info("=" * 55)
        run_categories(conn, dry_run=args.dry_run)

    if args.props:
        logger.info("=" * 55)
        logger.info("PASS: props")
        logger.info("=" * 55)
        run_props(conn, dry_run=args.dry_run)

    conn.close()
    logger.info("Enrichment complete.")


if __name__ == "__main__":
    main()
