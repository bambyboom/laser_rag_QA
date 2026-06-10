"""LLM-based relevance reranker for retrieved candidate chunks.

Provides :class:`Reranker` which uses an LLM (e.g. gpt-4o-mini or the
configured generation model) to score each candidate text chunk against
the original user query on a 1–10 relevance scale.  Chunks are then
re-ranked by that score, and only the top-*N* are kept for the
generation step.

Usage::

    from src.retrieval.reranker import Reranker
    from src.retrieval.retriever import HybridRetriever

    retriever = HybridRetriever(...)
    candidates = retriever.retrieve("YAG激光器怎么选型", top_k=15)

    reranker = Reranker()
    top_chunks = reranker.rerank("YAG激光器怎么选型", candidates, top_n=5)
    for cr in top_chunks:
        print(f"{cr.rerank_score}/10  {cr.text[:60]}...")
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from openai import OpenAI

if TYPE_CHECKING:
    from config import Settings
    from src.retrieval.retriever import RetrievalResult

logger = logging.getLogger("laser-rag.reranker")

# ---------------------------------------------------------------------------
#  output dataclass
# ---------------------------------------------------------------------------


@dataclass
class RerankerResult:
    """A retrieved chunk with its LLM-assigned relevance score.

    Attributes
    ----------
    rerank_score: LLM relevance rating, 1–10 (or 0 on parse failure).
    retrieval_score: Original fused score from the hybrid retriever.
    """

    id: str
    text: str
    filename: str = ""
    chunk_id: int = 0
    page: int | None = None
    rerank_score: int = 0          # 1–10 LLM relevance
    retrieval_score: float = 0.0   # original fused RRF score
    vector_score: float | None = None
    keyword_score: float | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_retrieval(
        cls, result: RetrievalResult, rerank_score: int
    ) -> RerankerResult:
        """Construct from a :class:`~src.retrieval.retriever.RetrievalResult`."""
        return cls(
            id=result.id,
            text=result.text,
            filename=result.filename,
            chunk_id=result.chunk_id,
            page=result.page,
            rerank_score=rerank_score,
            retrieval_score=result.score,
            vector_score=result.vector_score,
            keyword_score=result.keyword_score,
            metadata=dict(result.metadata),
        )


# ---------------------------------------------------------------------------
#  system prompt — scoring rubric
# ---------------------------------------------------------------------------

RERANK_SYSTEM_PROMPT = """\
你是一个激光器行业的知识检索评估专家。你的任务是根据用户问题，对多个候选文本片段逐一打分。

评分标准（1~10 整数）：

10 — 完美匹配：直接、完整地回答了用户问题，包含具体的参数、方法或数据
 9 — 几乎完美：回答了问题核心，仅缺少少量次要细节
 8 — 高度相关：提供了回答问题的关键信息，可直接用于生成答案
 7 — 较相关：包含了回答问题所需的大部分背景，但部分内容侧重点不同
 6 — 部分相关：涉及问题领域但未直接回答问题，可作为补充参考
 5 — 弱相关：同属一个大领域，但与具体问题关联不强
 4 — 仅有关键词重叠：包含个别相同的术语但讨论的是不同话题
 3 — 基本无关：与问题无实质关联
 2 — 明显无关：仅有一两个词碰巧相同
 1 — 完全不相关：与用户问题毫无关系

打分时请注意：
- 优先级：直接回答问题 > 提供背景知识 > 仅术语相同 > 完全无关
- 激光器行业术语要正确理解（如 YAG = Nd:YAG 固体激光器、LD = 半导体泵浦等）
- 包含具体参数（功率、波长、脉宽、频率）的片段应获更高分
- 片段长度不影响评分，只看内容相关性"""


# ===========================================================================
#  Reranker
# ===========================================================================


class Reranker:
    """LLM-powered relevance reranker for retrieved chunks.

    Parameters
    ----------
    model:       Model name for scoring (default: configured LLM model).
                 Using a cheaper / faster model like ``gpt-4o-mini`` is
                 recommended for this scoring pass.
    api_key:     API key (default from ``LLM_API_KEY`` config).
    api_base:    API base URL (default from ``LLM_API_BASE`` config).
    top_n:       Default number of chunks to keep after reranking
                 (overridable per call).
    max_chunk_chars: Truncate each chunk to this length before
                     sending to the LLM (saves tokens).
    temperature: LLM temperature — keep low for consistent scoring.
    timeout:     Per-request timeout in seconds.
    enabled:     If ``False``, ``rerank()`` returns the original
                 ranking unchanged (with rerank_score = 0).
    system_prompt: Custom system prompt for scoring.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        top_n: int | None = None,
        max_chunk_chars: int | None = None,
        temperature: float = 0.0,
        timeout: float | None = None,
        enabled: bool | None = None,
        system_prompt: str | None = None,
    ) -> None:
        from config import settings

        self._s: Settings = settings

        self.model = model or self._s.llm_model_name
        self.top_n = top_n if top_n is not None else self._s.reranker_top_n
        self.max_chunk_chars = (
            max_chunk_chars if max_chunk_chars is not None
            else self._s.reranker_max_chunk_chars
        )
        self.temperature = temperature
        self.timeout = timeout if timeout is not None else self._s.llm_timeout
        self.enabled = enabled if enabled is not None else self._s.reranker_enabled
        self.system_prompt = system_prompt or RERANK_SYSTEM_PROMPT

        _key = api_key or self._s.llm_api_key
        _base = api_base or self._s.llm_api_base

        if not _key:
            raise ValueError(
                "LLM API key not set. Provide it via LLM_API_KEY "
                "or OPENAI_API_KEY environment variable."
            )

        self._client = OpenAI(api_key=_key, base_url=_base, timeout=self.timeout)

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RerankerResult]:
        """Re-rank *candidates* by LLM relevance score for *query*.

        Parameters
        ----------
        query:      The original user question.
        candidates: Candidate chunks from the retriever (can be more
                    than *top_n* — the reranker will filter down).
        top_n:      Number of top results to keep (default from config).

        Returns
        -------
        list[RerankerResult]
            Sorted by ``rerank_score`` descending, then by original
            ``retrieval_score`` as tiebreaker.  Length ≤ *top_n*.

            When the reranker is disabled or the LLM call fails, the
            original ranking is preserved and ``rerank_score`` is set
            to 0.
        """
        if top_n is None:
            top_n = self.top_n

        if not candidates:
            return []

        if not self.enabled:
            logger.info("Reranker disabled — returning original order")
            return self._passthrough(candidates, top_n)

        # deduplicate by chunk id (keep first occurrence)
        seen: set[str] = set()
        deduped: list[RetrievalResult] = []
        for c in candidates:
            if c.id not in seen:
                seen.add(c.id)
                deduped.append(c)

        if len(deduped) == 0:
            return []

        try:
            scores = self._call_llm(query, deduped)
        except Exception:
            logger.exception("Reranker LLM call failed — returning original order")
            return self._passthrough(deduped, top_n)

        # build results with scores
        results = [
            RerankerResult.from_retrieval(chunk, scores.get(i, 0))
            for i, chunk in enumerate(deduped)
        ]

        # sort: rerank_score desc → retrieval_score desc as tiebreaker
        results.sort(
            key=lambda r: (r.rerank_score, r.retrieval_score),
            reverse=True,
        )

        kept = results[:top_n]
        logger.info(
            "Reranked %d → %d chunks (scores: %s)",
            len(deduped),
            len(kept),
            [r.rerank_score for r in kept],
        )
        return kept

    # ------------------------------------------------------------------
    #  internal — LLM call
    # ------------------------------------------------------------------
    def _call_llm(
        self, query: str, candidates: Sequence[RetrievalResult]
    ) -> dict[int, int]:
        """Score candidates via LLM → {candidate_index: score_1_to_10}."""
        user_prompt = self._build_user_prompt(query, candidates)

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=512,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content
        if not raw:
            logger.warning("Reranker LLM returned empty response")
            return {}

        return self._parse_response(raw, len(candidates))

    # ------------------------------------------------------------------
    #  prompt construction
    # ------------------------------------------------------------------
    def _build_user_prompt(
        self, query: str, candidates: Sequence[RetrievalResult]
    ) -> str:
        """Format query + numbered chunks for the LLM."""
        parts: list[str] = [
            f"用户问题：{query}",
            "",
            "请为以下每个候选片段打分（1~10 整数），输出格式：编号: 分数",
            "",
        ]

        for i, chunk in enumerate(candidates):
            text = chunk.text
            if len(text) > self.max_chunk_chars:
                text = text[: self.max_chunk_chars] + "…"
            # show source info for context
            source = chunk.filename or "未知来源"
            parts.append(
                f"--- 片段 {i}  [{source}] ---\n{text}"
            )

        parts.append("\n请输出每个片段的评分（每行一个）：")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    #  response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(raw: str, expected_count: int) -> dict[int, int]:
        """Parse LLM scoring output → ``{index: score}``.

        Handles formats like::

            0: 8
            1: 5
            片段 0: 8
            第0个片段：7分
            片段0：8

        Invalid / missing scores are silently dropped.
        """
        scores: dict[int, int] = {}

        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # ---- extract index ----
            idx: int | None = None

            # patterns: "0: 8", "片段 0: 8", "片段0：8分", "第0个：7"
            idx_match = re.search(r"(?:片段\s*|第\s*)?(\d+)\s*(?:个(?:片段)?)?\s*[:：]\s*(\d+)", line)
            if idx_match:
                try:
                    idx = int(idx_match.group(1))
                except ValueError:
                    continue
            else:
                # plain "数字: 数字" anywhere
                m2 = re.search(r"(?:^|\s)(\d+)\s*[:：]\s*(\d+)", line)
                if m2:
                    try:
                        idx = int(m2.group(1))
                    except ValueError:
                        continue

            if idx is None:
                continue

            # ---- extract score ----
            score_match = re.search(r"[:：]\s*(\d+)", line)
            if not score_match:
                continue

            try:
                score = int(score_match.group(1))
            except ValueError:
                continue

            # clamp to 1–10
            score = max(1, min(10, score))

            if 0 <= idx < expected_count:
                # keep the *first* score for each index (in case of duplicates)
                if idx not in scores:
                    scores[idx] = score

        return scores

    # ------------------------------------------------------------------
    #  fallback: disabled or error → passthrough
    # ------------------------------------------------------------------
    @staticmethod
    def _passthrough(
        candidates: Sequence[RetrievalResult], top_n: int
    ) -> list[RerankerResult]:
        """Return candidates in original order with rerank_score = 0."""
        return [
            RerankerResult.from_retrieval(c, rerank_score=0)
            for c in candidates[:top_n]
        ]
