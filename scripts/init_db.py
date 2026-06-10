#!/usr/bin/env python3
"""Initialise / manage the ChromaDB vector-store collection.

Usage::

    python scripts/init_db.py              # create collection if missing
    python scripts/init_db.py --reset      # delete & recreate collection
    python scripts/init_db.py --stats      # print collection info
    python scripts/init_db.py --clear      # remove all documents (keep schema)

Environment variables (see ``.env.example``):
    CHROMA_PERSIST_DIR     — vector-store path (default: ./chroma_db)
    CHROMA_COLLECTION_NAME — collection name  (default: laser_knowledge)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.knowledge.store import VectorStore

logger = logging.getLogger("laser-rag.scripts.init_db")


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _print_banner(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def _format_count(n: int) -> str:
    return f"{n:,} chunk{'s' if n != 1 else ''}"


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the Laser RAG QA ChromaDB collection",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the entire collection and recreate it from scratch",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove all documents but keep the collection schema",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print collection statistics and exit",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts (useful in scripts)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    store = VectorStore()
    collection_name = store._collection.name
    persist_dir = store._s.chroma_persist_dir

    # ------------------------------------------------------------------
    #  --stats  (informational, no mutation)
    # ------------------------------------------------------------------
    if args.stats:
        _print_banner("Collection Statistics")
        print(f"  Collection name : {collection_name}")
        print(f"  Persist dir     : {persist_dir}")
        print(f"  Document count  : {_format_count(store.count())}")
        print(f"  Indexed files   : {len(store.get_filenames())}")
        if store.count() > 0:
            print(f"\n  Files:")
            for fn in store.get_filenames():
                print(f"    • {fn}")
        return

    # ------------------------------------------------------------------
    #  --reset  (destroy & recreate)
    # ------------------------------------------------------------------
    if args.reset:
        if store.count() > 0 and not args.yes:
            _print_banner("⚠  WARNING")
            print(f"  This will permanently delete the collection")
            print(f"  {collection_name!r} ({_format_count(store.count())})")
            print(f"  at {persist_dir}")
            print()
            answer = input("  Type 'yes' to confirm: ")
            if answer.strip().lower() != "yes":
                print("  Aborted.")
                return

        before = store.count()
        store.reset()
        _print_banner("✓  Collection Reset")
        print(f"  Collection    : {collection_name!r}")
        print(f"  Chunks deleted: {_format_count(before)}")
        print(f"  Status        : empty, ready for ingest")
        logger.info("Reset complete — %s chunk(s) removed", _format_count(before))
        return

    # ------------------------------------------------------------------
    #  --clear  (remove docs, keep schema)
    # ------------------------------------------------------------------
    if args.clear:
        before = store.count()
        if before == 0:
            print("Collection is already empty — nothing to clear.")
            return

        if not args.yes:
            _print_banner("⚠  WARNING")
            print(f"  This will delete all {_format_count(before)} from")
            print(f"  {collection_name!r} (schema will be kept).")
            print()
            answer = input("  Type 'yes' to confirm: ")
            if answer.strip().lower() != "yes":
                print("  Aborted.")
                return

        store.clear()
        _print_banner("✓  Collection Cleared")
        print(f"  Chunks removed: {_format_count(before)}")
        print(f"  Remaining     : {_format_count(store.count())}")
        logger.info("Clear complete")
        return

    # ------------------------------------------------------------------
    #  default — ensure collection exists (no-op if already there)
    # ------------------------------------------------------------------
    _print_banner("Collection Ready")
    print(f"  Collection : {collection_name!r}")
    print(f"  Persist at : {persist_dir}")
    print(f"  Documents  : {_format_count(store.count())}")
    print(f"  Files      : {len(store.get_filenames())}")
    print()
    print("Use --reset to recreate, --clear to empty, --stats for details.")


if __name__ == "__main__":
    main()
