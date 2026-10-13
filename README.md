# MagiMatch

An AI-assisted magic-literature discovery application. It combines natural-language intent parsing, embedding similarity, title matching, magician-style expansion, and generated explanations in a Gradio interface.

## How it works

1. Scraped catalog metadata is normalized into SQLite.
2. OpenAI embeddings represent trick descriptions.
3. A query is interpreted into topics, keywords, and named-magician preferences.
4. Title matches and cosine-similarity results are merged, with per-book and per-author limits.
5. Language-model calls generate result descriptions and commentary.

The included code uses pretrained OpenAI models and a retrieval/ranking pipeline. It does not demonstrate reinforcement-learning model training. The notebook contains retrieval and query experiments.

## Setup

Use Python 3.11 or newer in a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='your-key'
# Optional: export SERPER_API_KEY='your-key'
python app.py
```

The application needs the SQLite catalog and embedding files described in `config/config.yaml`. The current app includes a first-run data download routine. Catalog data and embeddings are not committed in this source repository. API calls may incur charges.

## Structure

- `app.py`: Gradio application and data initialization.
- `src/query_engine.py`: query parsing, retrieval, and ranking.
- `src/openai_client.py`: API adapter.
- `src/scraper.py`, `processor.py`, `enricher.py`, `embedder.py`: ingestion pipeline.
- `src/db.py`, `checkpoint.py`, `config_loader.py`: persistence and configuration.
- `notebooks/magimatch-experiments.ipynb`: original experiments with saved outputs cleared.
- `tests/`: offline regression checks.

## Source selection

The existing GitHub repository (`KellaVK/MagiMatch`, commit `19d0362`) contains an older v4 snapshot and a scraper LFS object that GitHub returns as missing. This repository therefore uses the complete local v5 application from `MagiMatch`, rather than publishing a broken LFS pointer. The original upstream repository remains unchanged.

## Test

```sh
pip install pytest
python -m pytest -q
```

Tests use synthetic data and mocked embedding calls; they do not contact OpenAI, Serper, or the live catalog.

## Development notes

This is an organized source snapshot prepared in October 2026. Existing production deployments are independent of this repository. Keep local credentials and runtime data out of Git.
