"""Question router — classifies user questions into three routes.

After retrieval and reranking produce scored chunks, the router decides
whether the question should be answered from the knowledge base alone,
refused outright, or routed to web search.

Route logic
-----------
1. **ROUTE_KB_ONLY**     — KB has sufficient, relevant content.
2. **ROUTE_REFUSE**      — Question is clearly out-of-domain.
3. **ROUTE_WEB_SEARCH**  — In-domain but KB is insufficient; fall back
                           to web search (or user explicitly requested it).

Usage::

    from src.pipeline.router import Route, Router

    router = Router()
    route = await router.classify(question, top_chunks, enable_web_search=False)
    if route == Route.ROUTE_KB_ONLY:
        ...
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from src.retrieval.reranker import RerankerResult

logger = logging.getLogger("laser-rag.router")


# ===========================================================================
#  Route enum
# ===========================================================================


class Route(enum.Enum):
    """Routing decision for a user question."""

    ROUTE_KB_ONLY = "kb_only"          # Rule 1: fully answer from KB
    ROUTE_REFUSE = "refuse"            # Rule 2: out-of-domain, refuse
    ROUTE_WEB_SEARCH = "web_search"    # Rule 3: needs web search


# ===========================================================================
#  Threshold dataclass
# ===========================================================================


@dataclass
class RouterThresholds:
    """Configurable numeric thresholds for the routing decision tree.

    All thresholds reference the reranker 1–10 scoring scale:
      10 = perfect match, 7 = highly relevant, 4 = partially relevant,
      1–3 = irrelevant.

    Attributes
    ----------
    kb_sufficient:
        Top-1 rerank score must be >= this to consider KB sufficient.
    relevant_chunk_min:
        Minimum rerank score for a chunk to count as "relevant".
    relevant_chunk_count:
        Minimum number of relevant chunks needed to mark KB sufficient.
    out_of_domain:
        Top-1 score <= this AND domain classifier says "no" → refuse.
    """

    kb_sufficient: int = 7
    relevant_chunk_min: int = 4
    relevant_chunk_count: int = 2
    out_of_domain: int = 3


# ===========================================================================
#  Domain classifier system prompt
# ===========================================================================

_DOMAIN_CLASSIFIER_PROMPT = """\
你是激光器行业问题分类专家。判断用户问题是否属于激光器（laser）行业范畴。

激光器行业包括但不限于以下领域：

**激光器类型**：
- 固体激光器：Nd:YAG、Yb:YAG、Ti:Sapphire、红宝石、Cr:LiSAF 等
- 气体激光器：CO₂、He-Ne、Ar⁺、Kr⁺、准分子（Excimer）、氦镉 等
- 半导体激光器：LD、VCSEL、EEL、量子阱激光器、DFB、DBR 等
- 光纤激光器：掺镱、掺铒、掺铥、拉曼光纤、PCF 激光器 等
- 其他：染料激光器、自由电子激光器、量子级联激光器（QCL）、微片激光器 等

**核心部件与材料**：
- 泵浦源（LD泵浦、灯泵浦）、增益介质、谐振腔、Q开关、锁模器
- 非线性晶体（LBO、BBO、KTP、PPLN）、光学薄膜、反射镜、输出镜
- 光纤耦合器、合束器、光栅、隔离器、环形器

**激光参数与性能指标**：
- 功率/能量、波长/线宽、脉宽/重复频率、光束质量（M²、BPP）
- 转换效率、阈值、发散角、偏振度、稳定性、噪声

**激光应用领域**：
- 工业加工：切割、焊接、打标、钻孔、清洗、增材制造、热处理
- 医疗：眼科（LASIK）、皮肤科、牙科、泌尿科（碎石）、光动力治疗
- 通信：光纤通信、自由空间光通信、Li-Fi
- 雷达与传感：LiDAR、激光测距、激光陀螺、气体传感
- 科研：光谱学、非线性光学、超快光学、量子光学、原子冷却
- 国防安全：激光武器、测距、制导、激光炫目
- 消费电子：激光投影、激光打印、激光扫描

请回答：以上问题是否属于激光器行业领域？
只输出一个词：yes 或 no 或 partial。
- yes：明确属于激光器行业
- no：完全不属于（如编程语言、金融理财、普通医疗诊断、娱乐八卦、天气等）
- partial：部分相关或难以判断

用户问题：{question}
分类结果："""


# ===========================================================================
#  DomainClassifier
# ===========================================================================


class DomainClassifier:
    """Lightweight LLM call to determine if a question is in the laser domain.

    Uses the same LLM configuration as the rest of the pipeline but with
    temperature=0 and minimal tokens — the only expected output is a
    single word: ``yes``, ``no``, or ``partial``.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | None = None,
        max_retries: int = 1,
    ) -> None:
        from config import settings

        self.model = model or settings.llm_model_name
        self.timeout = timeout or settings.llm_timeout

        _key = api_key or settings.llm_api_key
        _base = api_base or settings.llm_api_base

        self._client = AsyncOpenAI(
            api_key=_key,
            base_url=_base,
            timeout=self.timeout,
            max_retries=max_retries,
        )

    async def classify(self, question: str) -> str:
        """Classify the question domain.

        Returns
        -------
        str
            One of ``"yes"``, ``"no"``, or ``"partial"``.
            Falls back to ``"partial"`` on any error.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=128,  # enough for reasoning tokens + output
                messages=[
                    {
                        "role": "system",
                        "content": _DOMAIN_CLASSIFIER_PROMPT,
                    },
                    {"role": "user", "content": f"用户问题：{question}"},
                ],
            )
            raw = (response.choices[0].message.content or "").strip().lower()
            if raw in ("yes", "no", "partial"):
                return raw
            logger.warning(
                "Unexpected domain classifier output: %r — falling back to 'partial'",
                raw[:100],
            )
            return "partial"
        except Exception:
            logger.exception(
                "Domain classifier LLM call failed for %r", question[:60]
            )
            return "partial"  # safe fallback — favours helpfulness


# ===========================================================================
#  Router
# ===========================================================================


class Router:
    """Classifies questions into one of three routes.

    The decision is based on:
    1. Reranker scores (1–10 per chunk)
    2. Domain classifier (yes / no / partial)
    3. User's explicit ``enable_web_search`` flag

    Parameters
    ----------
    domain_classifier:
        Pre-built :class:`DomainClassifier`.  Created lazily when ``None``.
    thresholds:
        Numeric thresholds for the decision tree.  Reads from config when
        ``None``.
    """

    def __init__(
        self,
        domain_classifier: DomainClassifier | None = None,
        thresholds: RouterThresholds | None = None,
    ) -> None:
        self._domain_classifier = domain_classifier
        self._thresholds = thresholds

    # ------------------------------------------------------------------
    #  classify
    # ------------------------------------------------------------------

    async def classify(
        self,
        question: str,
        top_chunks: list[RerankerResult],
        enable_web_search: bool = False,
    ) -> Route:
        """Decide which route the question should take.

        Parameters
        ----------
        question:
            Original user question.
        top_chunks:
            Reranker results, sorted by ``rerank_score`` descending.
        enable_web_search:
            Whether the user explicitly requested web search mode.

        Returns
        -------
        Route
            The chosen route.
        """
        t = self._get_thresholds()

        # ---- no KB results at all ----
        if not top_chunks:
            domain = await self._get_domain_classifier().classify(question)
            if domain == "no":
                logger.info(
                    "Route REFUSE (no KB results, domain=no) — %r", question[:60]
                )
                return Route.ROUTE_REFUSE
            logger.info(
                "Route WEB_SEARCH (no KB results, domain=%s) — %r",
                domain, question[:60],
            )
            return Route.ROUTE_WEB_SEARCH

        top_score = top_chunks[0].rerank_score
        relevant_count = sum(
            1 for c in top_chunks if c.rerank_score >= t.relevant_chunk_min
        )

        # ---- reranker passthrough fallback ----
        # When all scores are 0 the reranker likely failed (passthrough).
        # Use retrieval result count as a heuristic: if we have >= 2
        # chunks, treat as KB_ONLY rather than auto-routing to web search.
        # Exception: if user explicitly requested web search, skip passthrough.
        all_zero = all(c.rerank_score == 0 for c in top_chunks)
        if all_zero and len(top_chunks) >= t.relevant_chunk_count:
            if not enable_web_search:
                logger.info(
                    "Route KB_ONLY (reranker passthrough, %d chunks) — %r",
                    len(top_chunks), question[:60],
                )
                return Route.ROUTE_KB_ONLY

        # ---- Rule 1: KB sufficient ----
        if top_score >= t.kb_sufficient and relevant_count >= t.relevant_chunk_count:
            logger.info(
                "Route KB_ONLY (top_score=%d, relevant=%d) — %r",
                top_score, relevant_count, question[:60],
            )
            return Route.ROUTE_KB_ONLY

        # ---- Rule 3: user explicitly requested web search ----
        if enable_web_search:
            logger.info(
                "Route WEB_SEARCH (user requested) — %r", question[:60]
            )
            return Route.ROUTE_WEB_SEARCH

        # ---- borderline: need domain classification ----
        domain = await self._get_domain_classifier().classify(question)

        if top_score <= t.out_of_domain:
            if domain == "no":
                logger.info(
                    "Route REFUSE (top_score=%d, domain=no) — %r",
                    top_score, question[:60],
                )
                return Route.ROUTE_REFUSE
            logger.info(
                "Route WEB_SEARCH (top_score=%d, domain=%s) — %r",
                top_score, question[:60],
            )
            return Route.ROUTE_WEB_SEARCH

        # Score 4–6: partially relevant, in-domain → web search
        logger.info(
            "Route WEB_SEARCH (top_score=%d, relevant=%d) — %r",
            top_score, relevant_count, question[:60],
        )
        return Route.ROUTE_WEB_SEARCH

    # ------------------------------------------------------------------
    #  lazy initialisers
    # ------------------------------------------------------------------

    def _get_domain_classifier(self) -> DomainClassifier:
        if self._domain_classifier is None:
            self._domain_classifier = DomainClassifier()
        return self._domain_classifier

    def _get_thresholds(self) -> RouterThresholds:
        if self._thresholds is None:
            from config import settings

            self._thresholds = RouterThresholds(
                kb_sufficient=settings.rerank_kb_sufficient_threshold,
                relevant_chunk_min=settings.rerank_relevant_chunk_min,
                relevant_chunk_count=settings.rerank_relevant_chunk_count,
                out_of_domain=settings.rerank_out_of_domain_threshold,
            )
        return self._thresholds
