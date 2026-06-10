from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger("laser-rag.rewriter")

# ---------------------------------------------------------------------------
#  system prompt with few-shot laser-industry examples
# ---------------------------------------------------------------------------

REWRITE_SYSTEM_PROMPT = """\
你是一个激光器行业的知识检索专家。你的任务是将用户提出的自然语言问题改写成 2~3 个更短、更适合向量检
索的查询短语。要求:

1. 每个查询 8~25 个字,使用关键词组合而非自然语句。
2. 展开行业缩写和俗称,例如:
   - "YAG" → "Nd:YAG 固体激光器"
   - "LD" → "半导体激光二极管泵浦"
   - "MOPA" → "主振荡功率放大"
   - "QCW" → "准连续波"
   - "DPSS" → "二极管泵浦固体激光器"
3. 补充同义词和相关术语,覆盖不同角度的检索需求。
4. 保留原文中的关键参数(如波长、功率、脉宽等数值)。
5. 只输出改写后的查询,每行一个,不要编号、不要前缀、不要额外解释。

Examples:
用户: YAG激光器选型要注意什么
Nd:YAG 固体激光器 选型 参数 注意事项
YAG 激光器 规格 选型指南
掺钕钇铝石榴石 激光器 选型

用户: 光纤激光器功率一般多大
光纤激光器 输出功率 范围
脉冲光纤激光器 连续光纤激光器 功率对比
光纤激光器 功率 参数 规格

用户: 激光器泵浦方式有哪些
激光器 泵浦方式 LD泵浦 闪光灯泵浦
半导体激光二极管 泵浦 优缺点
固体激光器 泵浦技术 类型

用户: 光束质量怎么评估
激光器 光束质量 M²因子 评估
光束质量 BPP 测量 方法
高光束质量 激光器 参数 指标

用户: 1064nm激光器应用
1064nm 波长 激光器 工业应用
Nd:YAG 激光器 1064nm 加工领域
近红外激光器 1064nm 应用场景

用户: 紫外激光器用途
紫外激光器 准分子 固体紫外 应用
UV激光 精密加工 微电子
355nm 266nm 紫外激光 用途"""


# ---------------------------------------------------------------------------
#  QueryRewriter
# ---------------------------------------------------------------------------


class QueryRewriter:
    """Expand a natural-language question into multiple retrieval-friendly queries.

    Uses the LLM to perform terminology expansion, synonym injection,
    and abbreviation resolution specific to the laser industry.

    Usage::

        rewriter = QueryRewriter()
        queries = rewriter.rewrite("YAG激光器怎么选型")
        # queries → [
        #     "YAG激光器怎么选型",           # original (always first)
        #     "Nd:YAG 固体激光器 选型 参数",
        #     "YAG 激光器 规格 选型指南",
        # ]
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        system_prompt: str | None = None,
    ) -> None:
        from config import settings

        self._s: Settings = settings

        self.model = model or self._s.llm_model_name
        self.temperature = (
            temperature if temperature is not None else self._s.llm_temperature
        )
        self.max_tokens = (
            max_tokens if max_tokens is not None else self._s.llm_max_tokens
        )
        self.timeout = timeout if timeout is not None else self._s.llm_timeout
        self.system_prompt = system_prompt or REWRITE_SYSTEM_PROMPT

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
    def rewrite(self, query: str) -> list[str]:
        """Rewrite *query*, returning ``[original, variant_1, variant_2, ...]``.

        The original query is always retained as the first element so
        it can serve as a fallback if all rewrites fail to retrieve.
        """
        if not query.strip():
            return []

        try:
            variants = self._call_llm(query)
        except Exception:
            logger.exception("Query rewriting failed — falling back to original")
            return [query]

        # deduplicate while preserving order
        seen: set[str] = {query}
        result = [query]
        for v in variants:
            if v and v not in seen:
                result.append(v)
                seen.add(v)

        logger.info("Query rewritten: %r → %d variants", query[:60], len(result) - 1)
        return result

    def rewrite_only(self, query: str) -> list[str]:
        """Like :meth:`rewrite` but excludes the original query."""
        all_queries = self.rewrite(query)
        return all_queries[1:] if len(all_queries) > 1 else []

    # ------------------------------------------------------------------
    #  internal — LLM call + response parsing
    # ------------------------------------------------------------------
    def _call_llm(self, query: str) -> list[str]:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"用户: {query}"},
            ],
        )

        raw = response.choices[0].message.content
        if not raw:
            logger.warning("LLM returned empty rewrite for %r", query)
            return []

        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> list[str]:
        """Parse LLM output: one query per line, strip numbering/bullets."""
        queries: list[str] = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # remove common prefixes: "1.", "-", "•", "·", "→", "查询:", etc.
            line = re.sub(
                r"^(\d+[.、．)\-]\s*|[-•·→➜]\s*|查询\d*\s*[:：]\s*)+",
                "",
                line,
            ).strip()
            if line and len(line) >= 2:
                queries.append(line)
        return queries
