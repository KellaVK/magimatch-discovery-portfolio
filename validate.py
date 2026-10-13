"""
validate.py — MagiMatch v5 validation. Run after each step.

Usage:
    PYTHONPATH=. python validate.py --step 1   # config + DB schema
    PYTHONPATH=. python validate.py --step 2   # raw JSON batches
    PYTHONPATH=. python validate.py --step 3   # SQLite database
    PYTHONPATH=. python validate.py --step 4   # embeddings
    PYTHONPATH=. python validate.py --all
"""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

passed = 0
failed = 0


def check(label, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {label}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        failed += 1


def section(title):
    print(f"\n── {title} {'─'*(50-len(title))}")


# ── Step 1: Scaffold ───────────────────────────────────────────────────────────
def validate_step1():
    section("Step 1: Scaffold")

    def test_config():
        from src.config_loader import cfg, paths

        assert cfg.project.name == "MagiMatch"
        assert hasattr(paths, "db")

    check("config.yaml loads", test_config)

    def test_schema():
        import tempfile
        from src.db import get_connection, init_schema

        tmp = Path(tempfile.mktemp(suffix=".db"))
        init_schema(tmp)
        conn = get_connection(tmp)
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for t in ["books", "persons", "tricks", "book_persons"]:
            assert t in tables, f"Missing table: {t}"
        # Check lean schema — should NOT have the old junk columns
        cols = {r[1] for r in conn.execute("PRAGMA table_info(books)")}
        assert "isbn" not in cols, "books.isbn should not exist"
        assert "subtitle" not in cols, "books.subtitle should not exist"
        conn.close()
        tmp.unlink()

    check("DB schema is lean (no junk columns)", test_schema)

    def test_checkpoint():
        import tempfile
        from src.checkpoint import Checkpoint

        cp = Checkpoint("test", Path(tempfile.mkdtemp()))
        cp.mark_done("x")
        assert cp.is_done("x")
        cp.reset()
        assert not cp.is_done("x")

    check("Checkpoint works", test_checkpoint)


# ── Step 2: Raw data ───────────────────────────────────────────────────────────
def validate_step2():
    section("Step 2: Raw JSON batches")
    from src.config_loader import paths

    def test_files_exist():
        files = list(Path(paths.raw_data).glob("batch_*.json"))
        assert len(files) > 0, f"No batch files in {paths.raw_data}"
        print(f"     Found {len(files)} batch files")

    check("Batch files exist", test_files_exist)

    def test_structure():
        files = sorted(Path(paths.raw_data).glob("batch_*.json"))
        with open(files[0]) as f:
            books = json.load(f)
        assert isinstance(books, list) and len(books) > 0
        b = books[0]
        for field in ["archive_id", "title", "tricks"]:
            assert field in b, f"Missing field: {field}"
        if b["tricks"]:
            t = b["tricks"][0]
            for field in ["title", "categories", "creators"]:
                assert field in t, f"Trick missing field: {field}"

    check("Book/trick structure is correct", test_structure)

    def test_coverage():
        files = sorted(Path(paths.raw_data).glob("batch_*.json"))
        total_books = total_tricks = 0
        for f in files:
            books = json.load(open(f))
            total_books += len(books)
            total_tricks += sum(len(b.get("tricks", [])) for b in books)
        print(f"     {total_books} books, {total_tricks} tricks in raw data")
        assert total_books > 1000, f"Expected 3000+ books, got {total_books}"

    check("Coverage looks reasonable", test_coverage)


# ── Step 3: Database ───────────────────────────────────────────────────────────
def validate_step3():
    section("Step 3: SQLite database")
    from src.config_loader import paths
    from src.db import get_connection

    def test_db_exists():
        assert paths.db.exists(), f"DB not found: {paths.db}"

    check("DB file exists", test_db_exists)

    conn = get_connection(paths.db)

    def test_counts():
        b = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        t = conn.execute("SELECT COUNT(*) FROM tricks").fetchone()[0]
        p = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        print(f"     {b} books | {t} tricks | {p} persons")
        assert b >= 1000, f"Expected 3000+ books, got {b}"
        assert t >= 50000, f"Expected 100K+ tricks, got {t}"

    check("Row counts are healthy", test_counts)

    def test_no_junk():
        junk = ["Foreword", "Introduction", "Preface", "Conclusion"]
        for j in junk:
            n = conn.execute(
                "SELECT COUNT(*) FROM tricks WHERE LOWER(title)=LOWER(?)", (j,)
            ).fetchone()[0]
            assert n == 0, f"Found {n} junk tricks with title '{j}'"

    check("No front/back matter in tricks", test_no_junk)

    def test_embed_text():
        total = conn.execute("SELECT COUNT(*) FROM tricks").fetchone()[0]
        missing = conn.execute(
            "SELECT COUNT(*) FROM tricks WHERE embed_text IS NULL OR embed_text=''"
        ).fetchone()[0]
        pct = (total - missing) / total * 100
        print(f"     embed_text: {pct:.1f}% populated")
        assert pct > 95

    check("embed_text >95% populated", test_embed_text)

    def test_props():
        total = conn.execute("SELECT COUNT(*) FROM tricks").fetchone()[0]
        with_props = conn.execute(
            "SELECT COUNT(*) FROM tricks WHERE props IS NOT NULL"
        ).fetchone()[0]
        pct = with_props / total * 100
        print(f"     props: {pct:.1f}% populated")
        # Expect at least 50% — categories drive this
        assert pct > 40

    check("props column populated for majority", test_props)

    def test_top_persons():
        rows = conn.execute("""
            SELECT name, trick_count FROM persons
            ORDER BY trick_count DESC LIMIT 5
        """).fetchall()
        print("     Top credited persons:")
        for r in rows:
            print(f"       {r['name']}: {r['trick_count']}")
        assert any(
            "Vernon" in r["name"] or "Marlo" in r["name"] or "Colombini" in r["name"]
            for r in rows
        )

    check("Top persons look right (Vernon/Marlo/etc)", test_top_persons)

    conn.close()


# ── Step 4: Embeddings ─────────────────────────────────────────────────────────
def validate_step4():
    section("Step 4: Embeddings")
    import numpy as np
    from src.config_loader import paths

    def test_files():
        assert (paths.embeddings / "tricks.npy").exists()
        assert (paths.embeddings / "trick_ids.npy").exists()

    check("tricks.npy and trick_ids.npy exist", test_files)

    def test_shape():
        embs = np.load(str(paths.embeddings / "tricks.npy"))
        ids = np.load(str(paths.embeddings / "trick_ids.npy"))
        print(f"     Embeddings shape: {embs.shape}")
        print(f"     IDs shape: {ids.shape}")
        assert embs.shape[1] == 1536, f"Expected 1536 dims, got {embs.shape[1]}"
        assert len(embs) == len(ids), "Embeddings and IDs length mismatch"
        assert len(embs) > 50000

    check("Embeddings shape and count are correct", test_shape)


# ── Runner ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    steps = [1, 2, 3, 4] if args.all else [args.step] if args.step else [1]

    for s in steps:
        {
            "1": validate_step1,
            "2": validate_step2,
            "3": validate_step3,
            "4": validate_step4,
        }[str(s)]()

    print(f"\n{'='*52}")
    print(f"Result: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
