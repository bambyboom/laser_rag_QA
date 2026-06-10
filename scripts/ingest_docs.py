#!/usr/bin/env python3
"""Batch-ingest all documents from ``data/raw/`` into the vector store.

Usage::

    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --dir /path/to/docs
    python scripts/ingest_docs.py --reset   # clear collection first

Environment variables (see ``.env.example``):
    EMBEDDING_API_KEY  — required for the embedding service
    CHROMA_PERSIST_DIR — vector-store path (default: ./chroma_db)
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

from src.pipeline.ingest import IngestPipeline
from src.knowledge.store import VectorStore

logger = logging.getLogger("laser-rag.scripts.ingest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-ingest documents into the laser-rag-qa vector store",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory containing source documents (default: data/raw/)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the vector-store collection before ingesting",
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

    # optional reset
    if args.reset:
        store = VectorStore()
        logger.info("Resetting collection (count was %d)", store.count())
        store.reset()

    # run pipeline
    pipeline = IngestPipeline()
    count = pipeline.run(raw_dir=args.dir)

    if count == 0:
        logger.warning("No chunks were stored — check your data/raw/ directory")
        sys.exit(1)

    logger.info("Ingest complete — %d chunk(s) now in vector store", count)


if __name__ == "__main__":
    main()
