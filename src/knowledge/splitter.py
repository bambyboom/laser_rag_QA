from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger("laser-rag.splitter")


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A single chunk produced by TextSplitter."""

    text: str
    filename: str = ""
    chunk_id: int = 0
    page: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
#  TextSplitter
# ---------------------------------------------------------------------------


class TextSplitter:
    """Recursive character text splitter, Chinese-friendly.

    Attempts to split on natural boundaries in priority order
    (paragraph → line → sentence → word → character), falling
    back to force-split only as a last resort.

    Usage::

        splitter = TextSplitter()
        chunks = splitter.split(text, filename="manual.pdf")
        # or batch:
        chunks = splitter.split_batch(loader.load_all())
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
        detect_pages: bool = True,
    ) -> None:
        from config import settings

        self._s: Settings = settings

        self.chunk_size = chunk_size if chunk_size is not None else self._s.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else self._s.chunk_overlap
        )
        self.separators = separators or list(self._s.chunk_separators)
        self.detect_pages = detect_pages

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------
    def split(
        self,
        text: str,
        filename: str = "",
        extra_metadata: dict | None = None,
    ) -> list[TextChunk]:
        """Split *text* into chunks with metadata.

        Parameters
        ----------
        text: The full document text.
        filename: Attached to every chunk for traceability.
        extra_metadata: Merged into the ``metadata`` dict of every chunk.
        """
        if not text.strip():
            logger.warning("Empty text for %s — no chunks produced", filename or "?")
            return []

        base_meta = dict(extra_metadata) if extra_metadata else {}

        # -- optional: split by page first, then chunk each page -------------
        if self.detect_pages and "\f" in text:
            pages = text.split("\f")
            chunks: list[TextChunk] = []
            for page_num, page_text in enumerate(pages, start=1):
                if not page_text.strip():
                    continue
                page_chunks = self._split_text(page_text)
                for c in page_chunks:
                    c.filename = filename
                    c.page = page_num
                    c.metadata.update(base_meta)
                chunks.extend(page_chunks)
            # renumber globally
            for i, c in enumerate(chunks):
                c.chunk_id = i
            return chunks

        # -- no page breaks: chunk the whole text ---------------------------
        raw_chunks = self._split_text(text)
        for i, c in enumerate(raw_chunks):
            c.chunk_id = i
            c.filename = filename
            c.metadata.update(base_meta)
        return raw_chunks

    def split_batch(
        self, docs: list[dict[str, str]]
    ) -> list[TextChunk]:
        """Split every document in a loader/cleaner-style dict list."""
        all_chunks: list[TextChunk] = []
        for doc in docs:
            fname = doc.get("filename", "")
            chunks = self.split(doc["text"], filename=fname)
            all_chunks.extend(chunks)
        logger.info(
            "Split %d doc(s) into %d chunk(s) (size=%d, overlap=%d)",
            len(docs),
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

    # ------------------------------------------------------------------
    #  core algorithm
    # ------------------------------------------------------------------
    def _split_text(self, text: str) -> list[TextChunk]:
        """Recursively split *text* and merge into sized chunks."""
        splits = self._recursive_split(text, self.separators)
        merged = self._merge_splits(splits)
        return [TextChunk(text=t) for t in merged]

    def _recursive_split(
        self, text: str, separators: list[str]
    ) -> list[str]:
        """Split *text* by *separators[0]*, recurse on oversize pieces."""
        if not separators:
            # final fallback: force-split by character
            return self._force_split(text)

        sep = separators[0]
        remaining = separators[1:]

        parts = _split_keep_sep(text, sep)

        result: list[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # try next separator
                result.extend(self._recursive_split(part, remaining))
        return result

    def _force_split(self, text: str) -> list[str]:
        """Character-level split when all separators exhausted."""
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge small adjacent splits into chunks ≤ chunk_size, with overlap."""
        if not splits:
            return []

        chunks: list[str] = []
        current = ""

        for split in splits:
            candidate = current + split if current else split
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # start new chunk; if the split alone exceeds chunk_size
                # it means it came from force_split — keep as-is
                current = split

        if current:
            chunks.append(current)

        # apply overlap between consecutive chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped: list[str] = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = overlapped[-1]
                if len(prev) > self.chunk_overlap:
                    prefix = prev[-self.chunk_overlap :]
                    overlapped.append(prefix + chunks[i])
                else:
                    overlapped.append(chunks[i])
            return overlapped

        return chunks


# ---------------------------------------------------------------------------
#  helper: split while preserving the separator
# ---------------------------------------------------------------------------
def _split_keep_sep(text: str, separator: str) -> list[str]:
    """Split by *separator*, re-attaching it to each piece (except the last).

    ``""`` separator returns individual characters.
    """
    if separator == "":
        return list(text)

    parts = text.split(separator)
    if len(parts) == 1:
        return parts
    # re-attach separator to all but the final element
    return [p + separator for p in parts[:-1]] + [parts[-1]]
