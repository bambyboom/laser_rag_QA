from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger("laser-rag.cleaner")

# ---------------------------------------------------------------------------
#  Unicode character categories for the optional character-level filter
# ---------------------------------------------------------------------------
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x2E80, 0x2EFF),   # CJK Radicals Supplement
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
    (0xFE30, 0xFE4F),   # CJK Compatibility Forms
)

# Characters *always* preserved regardless of filter settings
_ALWAYS_KEEP: set[str] = {"\n", "\r", " "}


def _build_char_filter_pattern(
    keep_chinese: bool = True,
    keep_english: bool = True,
    keep_digits: bool = True,
    keep_punctuation: bool = True,
) -> re.Pattern[str]:
    """Build a regex that matches characters to be **kept**."""

    allowed: list[str] = list(_ALWAYS_KEEP)

    if keep_chinese:
        for lo, hi in _CJK_RANGES:
            allowed.append(rf"\u{lo:04X}-\u{hi:04X}")

    if keep_english:
        allowed.append(r"a-zA-Z")

    if keep_digits:
        allowed.append(r"0-9")

    if keep_punctuation:
        # Western + CJK punctuation commonly found in technical docs
        allowed.append(
            r"\.\,\;\:\?\!\¡\¿"
            r"\(\)\[\]\{\}\"\'\\"
            r"\-\–\—\_\@\#\$\%\^\&\*\+\=\/\|\<\>"
            r"\~\`\¡\¿\£\¥\€"
            r"、"  # 、
            r"。"  # 。
            r"，"  # ，
            r"；"  # ；
            r"："  # ：
            r"？"  # ？
            r"！"  # ！
            r"‘’“”"  # ''""
            r"《》"  # 《》
            r"（）"  # （）
            r"【】"  # 【】
            r"±×÷≤≥"  # ±×÷≤≥
            r"°μαβ"  # °μαβ (common in laser docs)
        )

    pattern = f"[^{''.join(allowed)}]"
    return re.compile(pattern)


# compiled once at import time since the defaults rarely change
_DEFAULT_FILTER_RE = _build_char_filter_pattern()


# ---------------------------------------------------------------------------
#  TextCleaner
# ---------------------------------------------------------------------------


@dataclass
class CleanStats:
    """Statistics collected during a cleaning run."""

    original_chars: int = 0
    cleaned_chars: int = 0
    lines_removed_noise: int = 0
    lines_removed_repeated: int = 0
    lines_removed_short: int = 0
    blank_groups_collapsed: int = 0

    @property
    def reduction_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return round(100 * (1 - self.cleaned_chars / self.original_chars), 1)


class TextCleaner:
    """Pipeline-based text cleaner for document preprocessing.

    Each cleaning step is an independently toggleable method.
    Instantiate once, then call ``clean()`` on every text block.

    Parameters
    ----------
    noise_patterns:
        Substrings to match when removing header/footer lines.
        A line containing any of these (case-insensitive) is removed.
    min_line_length:
        Drop lines shorter than this after all other cleaning.
    max_consecutive_blanks:
        Collapse sequences of blank lines down to this many.
    remove_repeated_lines:
        If True, auto-detect and remove lines that appear too
        frequently across the document (header/footer artifacts).
    repeated_line_ratio:
        A line appearing in more than this fraction of "pages"
        (estimated via form-feed or 50-line chunks) is considered noise.
    keep_chinese / keep_english / keep_digits / keep_punctuation:
        Character categories to retain during the optional character filter.
        All are on by default — set to False to strip that category.
    """

    def __init__(
        self,
        *,
        noise_patterns: list[str] | None = None,
        min_line_length: int | None = None,
        max_consecutive_blanks: int | None = None,
        remove_repeated_lines: bool | None = None,
        repeated_line_ratio: float | None = None,
        keep_chinese: bool = True,
        keep_english: bool = True,
        keep_digits: bool = True,
        keep_punctuation: bool = True,
    ) -> None:
        from config import settings

        self._s: Settings = settings

        self.noise_patterns = noise_patterns or list(self._s.cleaner_noise_patterns)
        self.min_line_length = (
            min_line_length
            if min_line_length is not None
            else self._s.cleaner_min_line_length
        )
        self.max_consecutive_blanks = (
            max_consecutive_blanks
            if max_consecutive_blanks is not None
            else self._s.cleaner_max_consecutive_blanks
        )
        self.remove_repeated_lines_flag = (
            remove_repeated_lines
            if remove_repeated_lines is not None
            else self._s.cleaner_remove_repeated_lines
        )
        self.repeated_line_ratio = (
            repeated_line_ratio
            if repeated_line_ratio is not None
            else self._s.cleaner_repeated_line_ratio
        )

        self.keep_chinese = keep_chinese
        self.keep_english = keep_english
        self.keep_digits = keep_digits
        self.keep_punctuation = keep_punctuation
        self._filter_re = _build_char_filter_pattern(
            keep_chinese, keep_english, keep_digits, keep_punctuation
        )

        # compile noise patterns into lowercase for case-insensitive matching
        self._noise_lower: list[str] = [p.lower() for p in self.noise_patterns if p]

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------
    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline on *text*."""
        stats = CleanStats(original_chars=len(text))
        text = self._normalize_line_endings(text)
        text = self._normalize_whitespace(text)
        text = self._strip_lines(text)
        text, stats.blank_groups_collapsed = self._collapse_blank_lines(text)
        text, stats.lines_removed_noise = self._remove_noise_lines(text)
        if self.remove_repeated_lines_flag:
            text, stats.lines_removed_repeated = self._remove_repeated_lines(text)
        text = self._filter_characters(text)
        text, stats.lines_removed_short = self._remove_short_lines(text)
        text = text.strip()
        stats.cleaned_chars = len(text)

        if stats.reduction_pct > 50:
            logger.warning(
                "High text reduction (%.1f%%). Verify noise patterns and filters.",
                stats.reduction_pct,
            )
        return text

    def clean_batch(
        self, docs: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Clean every document in *docs* (same shape as DocumentLoader output)."""
        cleaned: list[dict[str, str]] = []
        for doc in docs:
            original = doc["text"]
            result = self.clean(original)
            if not result.strip():
                logger.warning("Document %s is empty after cleaning — dropped", doc["filename"])
                continue
            cleaned.append({"filename": doc["filename"], "text": result})
        logger.info("Cleaned %d documents, %d retained", len(docs), len(cleaned))
        return cleaned

    # ------------------------------------------------------------------
    #  step 1 — line endings
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        """Convert ``\\r\\n`` and bare ``\\r`` to ``\\n``."""
        # \r\n → \n first, then any remaining \r → \n
        return text.replace("\r\n", "\n").replace("\r", "\n")

    # ------------------------------------------------------------------
    #  step 2 — whitespace
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Replace special whitespace characters with standard equivalents."""
        # tab → 4 spaces (but keep it simple: tab → space)
        text = text.replace("\t", " ")
        # non-breaking space (U+00A0) → regular space
        text = text.replace("\xa0", " ")
        # zero-width characters → remove
        for ch in ("​", "‌", "‍", "﻿", "‎", "‏"):
            text = text.replace(ch, "")
        # full-width space (U+3000) → regular space
        text = text.replace("　", " ")
        return text

    # ------------------------------------------------------------------
    #  step 3 — strip lines
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_lines(text: str) -> str:
        return "\n".join(line.strip() for line in text.split("\n"))

    # ------------------------------------------------------------------
    #  step 4 — blank lines
    # ------------------------------------------------------------------
    def _collapse_blank_lines(self, text: str) -> tuple[str, int]:
        """Collapse runs of blank lines to at most *max_consecutive_blanks*."""
        if self.max_consecutive_blanks < 1:
            # remove all blank lines
            lines = [line for line in text.split("\n") if line.strip()]
            return "\n".join(lines), 0

        collapsed: list[str] = []
        blank_run = 0
        count = 0
        for line in text.split("\n"):
            if not line.strip():
                blank_run += 1
                if blank_run <= self.max_consecutive_blanks:
                    collapsed.append("")
                else:
                    count += 1  # excess blank suppressed
            else:
                blank_run = 0
                collapsed.append(line)
        return "\n".join(collapsed), count

    # ------------------------------------------------------------------
    #  step 5 — pattern-based noise removal
    # ------------------------------------------------------------------
    def _remove_noise_lines(self, text: str) -> tuple[str, int]:
        """Drop lines that contain any substring in *noise_patterns*."""
        if not self._noise_lower:
            return text, 0

        kept: list[str] = []
        removed = 0
        for line in text.split("\n"):
            lower = line.lower()
            if any(pat in lower for pat in self._noise_lower):
                removed += 1
                logger.debug("Noise pattern removed: %r", line[:80])
                continue
            kept.append(line)
        return "\n".join(kept), removed

    # ------------------------------------------------------------------
    #  step 6 — repeated-line auto-detection (header / footer artifacts)
    # ------------------------------------------------------------------
    def _remove_repeated_lines(self, text: str) -> tuple[str, int]:
        """Remove lines that recur too often (likely headers/footers).

        "Pages" are estimated by counting form-feed characters or 50-line
        chunks.  A line that appears in more than ``repeated_line_ratio``
        fraction of pages is considered a header/footer artifact.
        """
        lines = text.split("\n")
        if len(lines) < 10:
            return text, 0

        # estimate page count
        ff_count = text.count("\f")
        if ff_count > 0:
            page_estimate = ff_count + 1
        else:
            page_estimate = max(1, len(lines) // 50)

        # count non-trivial lines
        non_empty = [line for line in lines if line.strip()]
        counter = Counter(non_empty)
        threshold = max(2, int(page_estimate * self.repeated_line_ratio))

        noise_set: set[str] = {
            line
            for line, freq in counter.items()
            if freq >= threshold and len(line) < 200
        }

        if not noise_set:
            return text, 0

        kept: list[str] = []
        removed = 0
        for line in lines:
            if line in noise_set:
                removed += 1
                logger.debug("Repeated line removed: %r", line[:80])
                continue
            kept.append(line)

        return "\n".join(kept), removed

    # ------------------------------------------------------------------
    #  step 7 — character-level filter
    # ------------------------------------------------------------------
    def _filter_characters(self, text: str) -> str:
        """Strip characters outside the allowed Unicode categories."""
        # skip if all categories are on (no filtering needed)
        if all([self.keep_chinese, self.keep_english, self.keep_digits, self.keep_punctuation]):
            return text
        return self._filter_re.sub("", text)

    # ------------------------------------------------------------------
    #  step 8 — short-line removal
    # ------------------------------------------------------------------
    def _remove_short_lines(self, text: str) -> tuple[str, int]:
        """Drop lines shorter than *min_line_length* (after stripping)."""
        if self.min_line_length <= 0:
            return text, 0

        kept: list[str] = []
        removed = 0
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and len(stripped) < self.min_line_length:
                removed += 1
                continue
            kept.append(line)
        return "\n".join(kept), removed


# ---------------------------------------------------------------------------
#  module-level convenience singleton
# ---------------------------------------------------------------------------
_default_cleaner: TextCleaner | None = None


def get_cleaner(**overrides) -> TextCleaner:
    """Return a cached default cleaner; pass kwargs to create a custom one."""
    global _default_cleaner
    if overrides:
        return TextCleaner(**overrides)
    if _default_cleaner is None:
        _default_cleaner = TextCleaner()
    return _default_cleaner
