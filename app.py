"""
MagiMatch — AI-powered magic literature discovery.
Hugging Face Spaces entry point.

On first boot this script:
  1. Clones the KellaVK/MagiMatch GitHub repo for source code (src/)
  2. Downloads the DB and embedding files from Google Drive
  3. Initializes the QueryEngine
  4. Launches the Gradio UI

Required Space secret: OPENAI_API_KEY
Optional Space secret: SERPER_API_KEY (for live web enrichment)
"""

import os
import sys
import subprocess
import random

# ─── 1. Pull source code from GitHub ──────────────────────────────────────────
REPO_NAME = "MagiMatch"
REPO_URL = f"https://github.com/KellaVK/{REPO_NAME}.git"

if not os.path.exists(REPO_NAME):
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL],
        check=True,
    )

# Make src/ importable and anchor cwd so relative data paths work
sys.path.insert(0, os.path.abspath(REPO_NAME))
os.chdir(REPO_NAME)
print(f"Working directory: {os.getcwd()}")


# ─── 2. Download data files from Google Drive ─────────────────────────────────
import gdown  # noqa: E402

DB_FILE_ID = "1s2XtNBtiXubmayBjFC1jY_Dt2WKhe17c"  # magimatch.db
EMB_FILE_ID = "1qzUzmDMolCvlBoaiXRGt-O7LCagpt9lX"  # tricks.npy
IDS_FILE_ID = "1OXz9hczKCAuw-Pl2D-R9XzagWiCOmeEm"  # trick_ids.npy

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/embeddings", exist_ok=True)

_files = [
    (DB_FILE_ID, "data/processed/magimatch.db", "magimatch.db"),
    (EMB_FILE_ID, "data/embeddings/tricks.npy", "tricks.npy"),
    (IDS_FILE_ID, "data/embeddings/trick_ids.npy", "trick_ids.npy"),
]

for file_id, dest, label in _files:
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print(f"  {label} already present")
    else:
        print(f"  Downloading {label} ...")
        gdown.download(id=file_id, output=dest, quiet=False)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  {label} ({size_mb:.1f} MB)")


# ─── 3. Initialize the query engine ───────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it under the Space's "
        "Settings → Variables and secrets."
    )

from src.query_engine import QueryEngine, build_result_card  # noqa: E402

engine = QueryEngine(
    db_path="data/processed/magimatch.db",
    embeddings_dir="data/embeddings",
    openai_api_key=api_key,
)
print("MagiMatch engine ready")


# ─── 4. Gradio UI ─────────────────────────────────────────────────────────────
import gradio as gr  # noqa: E402

BADGE_CONFIG = {
    "title_exact": ("EXACT TITLE", "#d4af37"),
    "title_fuzzy": ("TITLE MATCH", "#7eb8d4"),
    "semantic": ("SEMANTIC", "#9b8ec4"),
}


def render_card(card: dict, rank: int) -> str:
    badge_label, badge_color = BADGE_CONFIG.get(card["match_type"], ("MATCH", "#666"))
    category = (
        f'<span class="category-tag">{card["effect_category"]}</span>'
        if card["effect_category"]
        else ""
    )
    desc = card["ai_description"] or card["raw_description"]
    desc_html = f'<p class="card-desc">{desc}</p>' if desc else ""
    archive_link = (
        f'<a class="archive-link" href="{card["archive_url"]}" target="_blank" rel="noopener">'
        f"View on Conjuring Archive \u2197</a>"
        if card["archive_url"]
        else ""
    )
    return f"""
<div class="result-card">
  <div class="card-header">
    <div class="card-left">
      <span class="rank-num">#{rank}</span>
      <span class="trick-title">{card["trick_title"]}</span>
      {category}
    </div>
    <span class="match-badge" style="color:{badge_color};border-color:{badge_color}40">{badge_label}</span>
  </div>
  <div class="card-meta">
    <span class="credited-to">credited to <strong>{card["author"]}</strong></span>
    <span class="separator">\u00b7</span>
    <span class="book-title-meta">{card["book_title"]}</span>
    <span class="separator">\u00b7</span>
    <span class="pub-year">{card["pub_year"]}</span>
  </div>
  {desc_html}
  <div class="card-footer">{archive_link}</div>
</div>"""


def render_commentary(text: str) -> str:
    return f'<div class="commentary-block"><p>{text}</p></div>'


def render_enrichment_note(persons: list) -> str:
    if not persons:
        return ""
    return (
        f'<div class="enrichment-note">\U0001f310 Style profile fetched from web for: '
        f'{", ".join(persons)}</div>'
    )


def run_search(query: str):
    if not query or not query.strip():
        return "", '<p class="empty-state">Enter a search query above to begin.</p>'
    response = engine.search(query)
    results = response["results"]
    commentary = response["commentary"]
    web_enriched = response.get("web_enriched_persons", [])
    if not results:
        return (
            render_commentary("No results found for this query."),
            '<p class="empty-state">Try broader keywords or a different phrasing.</p>',
        )
    cards_html = "".join(
        render_card(build_result_card(r, i + 1), i + 1) for i, r in enumerate(results)
    )
    results_html = render_enrichment_note(web_enriched) + cards_html
    return render_commentary(commentary), results_html


SURPRISE_QUERIES = [
    "propless mentalism no objects",
    "coin magic pure sleight no gimmicks",
    "self-working card effects",
    "close-up rope magic",
    "memorized deck stacked",
    "stand-up comedy magic",
    "bizarre magic storytelling",
    "card to impossible location",
    "cups and balls technique",
    "gambling demonstrations false deals",
    "packet tricks visual",
    "silent act no talking",
    "book test mentalism",
    "ambitious card classic",
    "stage illusion design history",
]


def surprise_search():
    q = random.choice(SURPRISE_QUERIES)
    commentary, cards = run_search(q)
    return q, commentary, cards


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container, .main, .wrap {
    background: #0c0b09 !important;
    color: #e8e0d0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.app-header { padding: 32px 0 24px; border-bottom: 1px solid #2a2520; margin-bottom: 28px; }
.app-title { font-family: 'Playfair Display', Georgia, serif; font-size: 28px; font-weight: 700; color: #e8e0d0; letter-spacing: 0.02em; }
.app-subtitle { font-size: 12px; color: #6b6054; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 4px; }

.gr-textbox textarea, .gr-textbox input {
    background: #141210 !important; border: 1px solid #2a2520 !important;
    border-radius: 3px !important; color: #e8e0d0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-size: 14px !important;
    padding: 12px 16px !important; transition: border-color 0.2s;
}
.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: #d4af37 !important; outline: none !important;
    box-shadow: 0 0 0 2px #d4af3715 !important;
}
.gr-textbox label {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    color: #6b6054 !important; margin-bottom: 6px !important;
}

.gr-button {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    border-radius: 3px !important; transition: all 0.15s !important; cursor: pointer !important;
}
.gr-button.primary { background: #d4af37 !important; color: #0c0b09 !important; border: none !important; font-weight: 600 !important; }
.gr-button.primary:hover { background: #e8c84a !important; }
.gr-button.secondary { background: transparent !important; color: #9b8d7a !important; border: 1px solid #2a2520 !important; }
.gr-button.secondary:hover { border-color: #4a433a !important; color: #e8e0d0 !important; }

.gr-tab-nav button {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    color: #6b6054 !important; background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important; padding: 10px 16px !important; transition: all 0.15s !important;
}
.gr-tab-nav button.selected, .gr-tab-nav button:hover { color: #e8e0d0 !important; border-bottom-color: #d4af37 !important; }

.commentary-block { border-left: 3px solid #d4af37; padding: 14px 18px; background: #111009; margin-bottom: 20px; border-radius: 0 3px 3px 0; }
.commentary-block p { font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; line-height: 1.7; color: #c8bfb0; font-style: italic; }

.result-card { background: #111009; border: 1px solid #1e1c17; border-radius: 4px; padding: 18px 20px; margin-bottom: 10px; transition: border-color 0.2s; }
.result-card:hover { border-color: #2a2520; }
.card-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 8px; }
.card-left { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; flex: 1; }
.rank-num { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #3d3830; flex-shrink: 0; }
.trick-title { font-family: 'Playfair Display', Georgia, serif; font-size: 17px; font-weight: 600; color: #e8e0d0; line-height: 1.3; }
.category-tag { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #6b6054; letter-spacing: 0.08em; text-transform: uppercase; background: #1a1814; padding: 2px 7px; border-radius: 2px; white-space: nowrap; }
.match-badge { font-family: 'IBM Plex Mono', monospace; font-size: 9px; font-weight: 500; letter-spacing: 0.15em; text-transform: uppercase; border: 1px solid; padding: 3px 7px; border-radius: 2px; white-space: nowrap; flex-shrink: 0; }
.card-meta { font-size: 12px; color: #6b6054; margin-bottom: 10px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.card-meta strong { color: #9b8d7a; }
.book-title-meta { font-style: italic; color: #5a534a; }
.separator { color: #2a2520; }
.card-desc { font-size: 13px; line-height: 1.65; color: #a09585; margin-bottom: 12px; }
.card-footer { margin-top: 4px; }
.archive-link { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.08em; color: #4a7a9b; text-decoration: none; text-transform: uppercase; transition: color 0.15s; }
.archive-link:hover { color: #7eb8d4; }
.empty-state { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #3d3830; text-align: center; padding: 40px 0; letter-spacing: 0.05em; }
.enrichment-note { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #4a7a9b; letter-spacing: 0.05em; margin-bottom: 14px; }
.gradio-container > .main > .wrap > .gap { gap: 0 !important; }
footer { display: none !important; }
.svelte-1ed2p3z { display: none !important; }
"""

with gr.Blocks(title="MagiMatch", css=CSS) as demo:

    gr.HTML("""
    <div class="app-header">
      <div class="app-title">MagiMatch</div>
      <div class="app-subtitle">126,000+ effects · 3,630 books · AI-powered discovery</div>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("Search"):
            with gr.Row(equal_height=True):
                query_box = gr.Textbox(
                    label="What are you looking for?",
                    placeholder=(
                        '"propless mentalism, not Matt Mello"  ·  '
                        '"coin work, no gimmicks"  ·  '
                        '"Expert at the Card Table"'
                    ),
                    lines=1,
                    scale=6,
                )
                with gr.Column(scale=1, min_width=100):
                    search_btn = gr.Button("Search", variant="primary")
                    surprise_btn = gr.Button("Surprise", variant="secondary")

            commentary_out = gr.HTML()
            results_out = gr.HTML()

            search_btn.click(run_search, [query_box], [commentary_out, results_out])
            query_box.submit(run_search, [query_box], [commentary_out, results_out])
            surprise_btn.click(
                surprise_search, [], [query_box, commentary_out, results_out]
            )

        with gr.Tab("Browse by Effect"):
            effect_dd = gr.Dropdown(
                label="Effect type",
                choices=[
                    "Card Magic",
                    "Coin Magic",
                    "Mentalism",
                    "Memory Systems",
                    "Rope Magic",
                    "Cups & Balls",
                    "Gambling Demonstrations",
                    "Stand-up / Platform",
                    "Bizarre Magic",
                    "Book Tests",
                    "Packet Tricks",
                    "Children's Magic",
                    "Comedy Magic",
                ],
            )
            effect_btn = gr.Button("Browse", variant="secondary")
            effect_commentary = gr.HTML()
            effect_results = gr.HTML()
            effect_btn.click(
                lambda e: run_search(e),
                [effect_dd],
                [effect_commentary, effect_results],
            )

        with gr.Tab("Browse by Person"):
            person_box = gr.Textbox(
                label="Magician name",
                placeholder="Dai Vernon · Matt Mello · Luke Jermay · Ed Marlo …",
                lines=1,
            )
            gr.HTML(
                "<p style=\"font-family:'IBM Plex Mono',monospace;font-size:10px;"
                'color:#3d3830;letter-spacing:0.08em;margin:6px 0 16px">'
                "30+ MAGICIANS HAVE CURATED STYLE PROFILES. OTHERS USE WEB OR DB LOOKUP.</p>"
            )
            person_btn = gr.Button("Find Similar", variant="secondary")
            person_commentary = gr.HTML()
            person_results = gr.HTML()
            person_btn.click(
                lambda p: run_search(f"similar to {p}"),
                [person_box],
                [person_commentary, person_results],
            )
            person_box.submit(
                lambda p: run_search(f"similar to {p}"),
                [person_box],
                [person_commentary, person_results],
            )


if __name__ == "__main__":
    demo.launch()
