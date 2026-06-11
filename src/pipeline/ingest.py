from __future__ import annotations

import logging
from pathlib import Path

from src.knowledge.cleaner import TextCleaner, get_cleaner
from src.knowledge.embedder import EmbeddedChunk, Embedder
from src.knowledge.loader import DocumentLoader
from src.knowledge.ocr import OcrProcessor
from src.knowledge.splitter import TextChunk, TextSplitter
from src.knowledge.store import VectorStore

logger = logging.getLogger("laser-rag.ingest")

# ---------------------------------------------------------------------------
#  IngestPipeline
# ---------------------------------------------------------------------------


class IngestPipeline:
    """Orchestrate the full document-to-vectorstore pipeline.

    Usage::

        pipeline = IngestPipeline()
        count = pipeline.run()                # batch from data/raw/
        count = pipeline.ingest_file(path)    # single file
        count = pipeline.ingest_text(text, filename="api_upload.txt")
    """

    def __init__(self) -> None:
        self.loader = DocumentLoader()
        self.cleaner = get_cleaner()
        self.splitter = TextSplitter()
        self.embedder = Embedder()
        self.store = VectorStore()
        self._ocr: OcrProcessor | None = None

    # ------------------------------------------------------------------
    #  full batch (directory → store)
    # ------------------------------------------------------------------
    def run(self, raw_dir: Path | str | None = None) -> int:
        """Load, clean, split, embed, and store every document in *raw_dir*.

        Returns the total number of chunks stored.
        """
        loader = (
            DocumentLoader(raw_dir=raw_dir) if raw_dir else self.loader
        )

        # ---- 1. load ---------------------------------------------------
        docs = loader.load_all()
        if not docs:
            logger.warning("No documents found — aborting ingest")
            return 0
        logger.info("Step 1/6  loaded %d document(s)", len(docs))

        # ---- 1.5 OCR (detect scanned PDFs) ----------------------------
        ocr_raw_dir = Path(raw_dir) if raw_dir else loader._raw_dir
        docs = self._get_ocr().process_docs(docs, ocr_raw_dir)
        logger.info("Step 1.5/6 OCR processed")

        # ---- 2. clean --------------------------------------------------
        docs = self.cleaner.clean_batch(docs)
        if not docs:
            logger.warning("All documents empty after cleaning — aborting ingest")
            return 0
        logger.info("Step 2/6  cleaned → %d document(s)", len(docs))

        # ---- 3. split --------------------------------------------------
        chunks: list[TextChunk] = self.splitter.split_batch(docs)
        if not chunks:
            logger.warning("No chunks produced — aborting ingest")
            return 0
        logger.info("Step 3/6  split → %d chunk(s)", len(chunks))

        # ---- 4. embed --------------------------------------------------
        embedded: list[EmbeddedChunk] = self.embedder.embed_chunks(chunks)
        logger.info(
            "Step 4/6  embedded → %d vector(s) (dim=%d)",
            len(embedded),
            len(embedded[0].embedding) if embedded else 0,
        )

        # ---- 5. store --------------------------------------------------
        stored = self.store.add(embedded)
        logger.info("Step 5/6  stored %d chunk(s) in vectorstore", stored)
        return stored

    # ------------------------------------------------------------------
    #  single file
    # ------------------------------------------------------------------
    def ingest_file(self, file_path: Path | str) -> int:
        """Process a single file through the full pipeline and store it.

        Returns the number of chunks stored.
        """
        fp = Path(file_path)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")

        doc = self.loader.load_file(fp)

        # OCR scanned PDFs — use the file's parent directory
        docs = self._get_ocr().process_docs([doc], fp.parent)

        return self._process_docs(docs)

    # ------------------------------------------------------------------
    #  raw text (for API uploads)
    # ------------------------------------------------------------------
    def ingest_text(self, text: str, filename: str = "upload.txt") -> int:
        """Process raw text (e.g. from an API upload) and store it.

        Returns the number of chunks stored.
        """
        if not text.strip():
            logger.warning("Empty text — nothing to ingest")
            return 0

        doc = {"filename": filename, "text": text}
        return self._process_docs([doc])

    # ------------------------------------------------------------------
    #  shared post-load pipeline
    # ------------------------------------------------------------------
    def _process_docs(self, docs: list[dict[str, str]]) -> int:
        """Clean → split → embed → store for an already-loaded doc list."""
        docs = self.cleaner.clean_batch(docs)
        if not docs:
            return 0

        chunks = self.splitter.split_batch(docs)
        if not chunks:
            return 0

        embedded = self.embedder.embed_chunks(chunks)
        return self.store.add(embedded)

    # ------------------------------------------------------------------
    #  lazy initialisers
    # ------------------------------------------------------------------
    def _get_ocr(self) -> OcrProcessor:
        if self._ocr is None:
            from config import settings

            self._ocr = OcrProcessor(
                enabled=settings.ocr_enabled,
                force=settings.ocr_force,
                language=settings.ocr_language,
                dpi=settings.ocr_dpi,
                min_text_length=settings.ocr_min_text_length,
                tesseract_cmd=settings.ocr_tesseract_cmd,
                tessdata_prefix=settings.ocr_tessdata_prefix,
            )
        return self._ocr
