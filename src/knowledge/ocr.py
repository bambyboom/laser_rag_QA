"""OCR processor for scanned/image-based PDFs using Tesseract OCR.

Detects documents where ``pypdf`` text extraction returned insufficient
content (typical of scanned PDFs with no embedded text layer), renders
each page to an image, and runs Tesseract OCR to recover the text.

Uses ``pymupdf`` (fitz) for PDF-to-image rendering — no poppler system
dependency required.  Tesseract must be installed separately::

    sudo apt install tesseract-ocr tesseract-ocr-chi-sim

Usage::

    from src.knowledge.ocr import OcrProcessor

    ocr = OcrProcessor()
    docs = ocr.process_docs(docs, raw_dir)
    # docs with empty/scanned text replaced by OCR output
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("laser-rag.ocr")


# ===========================================================================
#  OcrProcessor
# ===========================================================================


class OcrProcessor:
    """Detect scanned PDFs and run Tesseract OCR to recover text.

    Only processes ``.pdf`` files.  Other formats (``.docx``, ``.txt``,
    ``.md``, ``.html``) are passed through unchanged.

    Parameters
    ----------
    enabled:
        If ``False``, ``process_docs()`` is a no-op.  Automatically set
        to ``False`` when Tesseract is not installed.
    force:
        If ``True``, OCR every PDF regardless of whether pypdf already
        extracted text.  Default is ``False`` (auto-detect).
    language:
        Tesseract language string, e.g. ``"chi_sim+eng"``.
    dpi:
        Resolution for page rendering before OCR.  Higher = better
        accuracy but slower.
    min_text_length:
        Threshold below which a document is considered "scanned" and
        routed to OCR (character count, stripped).  Also checks for
        garbled/undecodable text patterns.
    """

    def __init__(
        self,
        enabled: bool = True,
        force: bool = False,
        language: str = "chi_sim+eng",
        dpi: int = 300,
        min_text_length: int = 50,
        tesseract_cmd: str = "tesseract",
        tessdata_prefix: str = "",
    ) -> None:
        self.enabled = enabled
        self.force = force
        self.language = language
        self.dpi = dpi
        self.min_text_length = min_text_length
        self.tesseract_cmd = tesseract_cmd
        self.tessdata_prefix = tessdata_prefix

        # ---- try importing tesseract ---------------------------------
        if self.enabled:
            try:
                import pytesseract  # noqa: F401

                self._pytesseract_available = True
            except ImportError:
                logger.warning(
                    "pytesseract not installed — OCR disabled. "
                    "Install with: pip install pytesseract"
                )
                self._pytesseract_available = False
                self.enabled = False

            # configure pytesseract with custom binary path
            if self._pytesseract_available:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

            # check that the tesseract binary is reachable
            if self._pytesseract_available:
                import shutil
                import os

                # When a custom tesseract_cmd is provided, check it directly
                # rather than relying on PATH.
                cmd_to_check = self.tesseract_cmd
                if cmd_to_check == "tesseract":
                    cmd_to_check = shutil.which("tesseract") or "tesseract"

                if not os.path.isfile(cmd_to_check) and not shutil.which(cmd_to_check.split()[0]):
                    logger.warning(
                        "Tesseract binary not found at %r — OCR disabled. "
                        "Install with: sudo apt install tesseract-ocr tesseract-ocr-chi-sim",
                        self.tesseract_cmd,
                    )
                    self._pytesseract_available = False
                    self.enabled = False

            # set TESSDATA_PREFIX if provided
            if self._pytesseract_available and self.tessdata_prefix:
                import os

                os.environ.setdefault("TESSDATA_PREFIX", self.tessdata_prefix)
                logger.debug("TESSDATA_PREFIX set to %s", self.tessdata_prefix)

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------

    def process_docs(
        self, docs: list[dict[str, str]], raw_dir: Path
    ) -> list[dict[str, str]]:
        """Process a list of document records, OCR-ing scanned PDFs.

        Each record is a dict with keys ``"filename"`` and ``"text"``.
        For records where the text is insufficient (or ``force=True``),
        the original ``.pdf`` file is re-read from *raw_dir* and OCR'd.

        Parameters
        ----------
        docs:
            Document records from :class:`~src.knowledge.loader.DocumentLoader`.
        raw_dir:
            Directory containing the original source files.

        Returns
        -------
        list[dict]
            The same list with ``text`` fields replaced when OCR succeeds.
        """
        if not self.enabled:
            return docs

        for doc in docs:
            filename = doc.get("filename", "")
            if not filename.lower().endswith(".pdf"):
                continue

            text = doc.get("text", "")

            if not self.needs_ocr(text):
                continue

            pdf_path = raw_dir / filename
            if not pdf_path.exists():
                logger.warning(
                    "Cannot OCR %s — file not found at %s", filename, pdf_path
                )
                continue

            logger.info(
                "OCR triggered for %s (extracted %d chars, min=%d)",
                filename,
                len(text.strip()),
                self.min_text_length,
            )

            ocr_text = self.process_file(pdf_path)
            if ocr_text.strip():
                doc["text"] = ocr_text
                logger.info(
                    "OCR produced %d chars for %s", len(ocr_text), filename
                )
            else:
                logger.warning(
                    "OCR returned empty text for %s — keeping original", filename
                )

        return docs

    def process_file(self, pdf_path: Path) -> str:
        """Run full OCR pipeline on a single PDF file.

        1. Render each page to a PIL Image via ``pymupdf``.
        2. Run Tesseract on each page image.
        3. Join pages with ``[第 N 页]`` markers.

        Returns the combined text, or an empty string on failure.
        """
        try:
            import fitz  # pymupdf
            import pytesseract
        except ImportError as exc:
            logger.error("Missing OCR dependency: %s", exc)
            return ""

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.exception("Failed to open PDF with pymupdf: %s", pdf_path)
            return ""

        pages: list[str] = []
        total = len(doc)

        # log progress for large PDFs every 20 pages
        progress_interval = max(1, min(20, total // 10))

        for i in range(total):
            try:
                page = doc[i]
                # Render page to a pixmap (RGB image)
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image for pytesseract
                from PIL import Image

                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # OCR the page
                text = pytesseract.image_to_string(img, lang=self.language)

                if text.strip():
                    pages.append(f"[第 {i + 1} 页]\n{text.strip()}")
                else:
                    logger.debug("No OCR text on page %d of %s", i + 1, pdf_path.name)

            except Exception:
                logger.exception(
                    "OCR failed on page %d of %s", i + 1, pdf_path.name
                )
                continue

            if (i + 1) % progress_interval == 0:
                logger.info(
                    "OCR progress: %d/%d pages (%d%%) — %s",
                    i + 1, total, int((i + 1) / total * 100), pdf_path.name,
                )

        doc.close()

        if not pages:
            logger.warning("OCR produced no text for %s", pdf_path.name)
            return ""

        return "\n\n".join(pages)

    # ------------------------------------------------------------------
    #  detection
    # ------------------------------------------------------------------

    def needs_ocr(self, text: str) -> bool:
        """Return ``True`` if *text* looks like it came from a scanned PDF.

        Criteria (OR):
        1. ``force`` is ``True``.
        2. Stripped text length < ``min_text_length``.
        3. Text contains garbled single-byte characters (high ratio of
           non-CJK / non-ASCII isolated characters — typical of failed
           pypdf extraction on CJK scanned content).
        """
        if self.force:
            return True

        stripped = text.strip()
        if len(stripped) < self.min_text_length:
            return True

        # Garbled detection: if the text has lots of isolated Latin-1
        # replacement chars or C0 control chars, it's likely a failed
        # extraction from a Chinese scanned PDF.
        garbled_ratio = _garbled_ratio(stripped)
        if garbled_ratio > 0.30:
            logger.debug(
                "Text looks garbled (ratio=%.2f) — OCR will be attempted",
                garbled_ratio,
            )
            return True

        return False


# ===========================================================================
#  helper — garbled text detection
# ===========================================================================


def _garbled_ratio(text: str) -> float:
    """Estimate the fraction of characters that look like garbled text.

    For Chinese documents, pypdf extraction on scanned pages often
    produces strings of isolated C1 control bytes, private-use-area
    codepoints, or non-printable characters.

    Returns a value in [0, 1]; higher = more suspicious.
    """
    if not text:
        return 0.0

    garbled = 0
    total = 0

    for ch in text:
        if ch.isspace():
            continue
        total += 1
        cp = ord(ch)

        # C0 / C1 control chars (except common whitespace)
        if cp < 0x20 or (0x7F <= cp <= 0x9F):
            garbled += 1
        # Private Use Area
        elif 0xE000 <= cp <= 0xF8FF:
            garbled += 1
        # Replacement character
        elif cp == 0xFFFD:
            garbled += 1
        # Isolated high-surrogate-like bytes (unlikely in valid UTF-8)
        elif 0xD800 <= cp <= 0xDFFF:
            garbled += 1

    if total == 0:
        return 0.0

    return garbled / total
