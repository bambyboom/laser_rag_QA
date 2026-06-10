"""Structured prompt builder for the RAG generation step.

Provides :func:`build_prompt` which takes a user question, a list of
retrieved (and optionally reranked) context chunks, and optional chat
history, and returns a fully-formed ``list[dict]`` of messages ready
for the LLM chat-completions endpoint.

Usage::

    from src.generation.prompt import build_prompt
    from src.retrieval.reranker import Reranker

    reranker = Reranker()
    chunks = reranker.rerank(query, candidates, top_n=5)

    messages = build_prompt(
        question="YAG激光器输出功率多少瓦？",
        contexts=chunks,
        chat_history=session_history,   # optional
    )
    # → [{"role": "system", ...}, {"role": "user", ...}]
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence, Union

if TYPE_CHECKING:
    from config import Settings
    from src.retrieval.reranker import RerankerResult

logger = logging.getLogger("laser-rag.prompt")

# ---------------------------------------------------------------------------
#  default few-shot examples (laser industry)
# ---------------------------------------------------------------------------

LASER_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    # ---- Example 1: de-duplication + integration, no inline markers ----
    {
        "role": "user",
        "content": (
            "参考资料：\n"
            "--- [Nd:YAG激光器技术手册] ---\n"
            "Nd:YAG固体激光器的输出功率范围通常在10W至500W之间，"
            "其中工业加工常用100W~300W的连续输出。高功率型号可达1kW以上，"
            "但需要额外的冷却系统支持。\n\n"
            "--- [固体激光器选型指南] ---\n"
            "Nd:YAG激光器在工业中常用功率为100W-300W，高端型号配备水冷系统"
            "后可达千瓦级输出。\n\n"
            "用户问题：YAG激光器的典型输出功率是多少？"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Nd:YAG 固体激光器的输出功率在 **10W 至 1kW 以上**的宽广范围内，"
            "具体取决于设计目标和冷却方案：\n\n"
            "- **工业加工**：常用 100W–300W 连续输出\n"
            "- **高功率应用**：配备水冷系统后可达 1kW 以上\n\n"
            "两份文档对工业常用功率的描述一致（100W–300W），已合并引用。\n\n"
            "> 📚 参考文档：\n"
            "> 1. 《Nd:YAG激光器技术手册》\n"
            "> 2. 《固体激光器选型指南》"
        ),
    },
    # ---- Example 2: KB partial info → web supplement ----
    {
        "role": "user",
        "content": (
            "参考资料：\n"
            "--- [光纤激光器产品规格] ---\n"
            "掺镱光纤激光器的电光转换效率可达30%~40%，远高于传统固体激光器。"
            "其输出波长范围为1030nm~1090nm。\n\n"
            "用户问题：CO₂激光器和光纤激光器哪个更适合切割厚钢板？"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "**知识库部分**\n\n"
            "根据现有资料，关于光纤激光器已知：\n"
            "- 电光转换效率：30%–40%\n"
            "- 输出波长范围：1030nm–1090nm\n\n"
            "但知识库中缺少 CO₂激光器的性能参数以及两者在厚钢板切割方面的对比数据。\n\n"
            "🌐 网络补充\n\n"
            "CO₂激光器与光纤激光器在厚钢板切割的选择要点：\n\n"
            "| 维度 | CO₂激光器 | 光纤激光器 |\n"
            "|------|----------|------------|\n"
            "| 适合厚度 | ≥12mm 厚板 | ≤12mm 薄中板 |\n"
            "| 切割速度 | 厚板中速 | 薄板极快 |\n"
            "| 切面质量 | 厚板切面光滑 | 厚板易有毛刺 |\n"
            "| 运行成本 | 较高（气体消耗） | 较低 |\n\n"
            "**结论**：切割厚钢板（≥12mm）优先选 CO₂激光器；薄中板大批量加工选光纤激光器。\n\n"
            "> 📚 参考文档：\n"
            "> 1. 《光纤激光器产品规格》\n"
            "> 2. 🌐 《Wikipedia - Laser cutting》 — https://en.wikipedia.org/wiki/Laser_cutting\n"
            "> 3. 🌐 《TRUMPF - CO₂ vs Fiber laser》 — https://www.trumpf.com/en/solutions/applications/laser-cutting/\n"
            ">\n"
            "> ⚠️ 以上标注「🌐 网络补充」的内容来自广域知识，非本地知识库文档，"
            "URL 来自模型训练数据可能已失效，请核实后再用于关键决策。"
        ),
    },
    # ---- Example 3: multi-source synthesis, no inline markers ----
    {
        "role": "user",
        "content": (
            "参考资料：\n"
            "--- [紫外激光器技术白皮书] ---\n"
            "紫外激光器（355nm/266nm）具有波长短、光子能量高的特点，"
            "加工时热影响区极小，适合对热敏感材料的精密加工。\n\n"
            "--- [PCB加工工艺手册] ---\n"
            "在PCB钻孔工序中，紫外激光可实现孔径50μm以下的微孔加工，"
            "孔壁光滑无碳化，是HDI板生产的首选方案。\n\n"
            "用户问题：紫外激光器在PCB行业的应用和优势是什么？"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "紫外激光器在 PCB 行业的应用集中在高精密微加工环节，核心优势"
            "来自其短波长和高光子能量：\n\n"
            "**主要应用**\n"
            "- 微孔钻孔：紫外激光可加工孔径 **50μm 以下**的微孔，孔壁光滑无碳化\n"
            "- 适用场景：HDI（高密度互连）板生产的首选方案\n\n"
            "**核心优势**\n"
            "- 波长极短（355nm / 266nm），单光子能量高\n"
            "- 热影响区极小，适合热敏感材料加工\n"
            "- 加工精度高，对基材几乎无损伤\n\n"
            "两份文档分别从技术原理和工程应用两个角度描述，互为补充，"
            "此处已整合为完整结论。\n\n"
            "> 📚 参考文档：\n"
            "> 1. 《紫外激光器技术白皮书》\n"
            "> 2. 《PCB加工工艺手册》"
        ),
    },
    # ---- Example 4: KB completely irrelevant → full web supplement ----
    {
        "role": "user",
        "content": (
            "参考资料：\n"
            "（暂无参考文档）\n\n"
            "用户问题：2025年激光雷达在自动驾驶行业的最新进展有哪些？"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "⚠️ 知识库中未找到相关文档，以下回答来自广域知识（🌐 网络补充）：\n\n"
            "🌐 网络补充\n\n"
            "2025年激光雷达（LiDAR）在自动驾驶行业的主要进展：\n\n"
            "1. **固态化趋势加速**：MEMS 振镜方案成熟量产，OPA（光学相控阵）"
            "进入前装测试阶段，成本降至 $200–$500 区间\n"
            "2. **FMCW 测距方案**：调频连续波方案可同时获取距离和速度信息，"
            "抗干扰能力显著优于传统 ToF 方案，成为 L4 车型标配\n"
            "3. **感知融合**：激光雷达与 4D 毫米波雷达 + 视觉的端到端融合方案"
            "在多家主机厂量产落地\n"
            "4. **中国供应链崛起**：禾赛、速腾聚创等国产厂商全球市占率合计超 60%\n\n"
            "> 📚 参考文档：\n"
            "> 1. 🌐 《Wikipedia - Lidar》 — https://en.wikipedia.org/wiki/Lidar\n"
            "> 2. 🌐 《Hesai Technology - AT128》 — https://www.hesaitech.com/product/at128/\n"
            "> 3. 🌐 《RoboSense - Products》 — https://www.robosense.ai/en/products\n"
            ">\n"
            "> ⚠️ 以上回答全部来自广域知识（🌐 网络补充），非本地知识库文档，"
            "URL 来自模型训练数据可能已失效，请核实后再用于关键决策。"
        ),
    },
]

# ---------------------------------------------------------------------------
#  sentinel for optional arguments (must be defined before function signatures)
# ---------------------------------------------------------------------------
_SENTINEL = object()

# ---------------------------------------------------------------------------
#  main entry point
# ---------------------------------------------------------------------------


def build_prompt(
    question: str,
    contexts: Sequence[Union[str, RerankerResult]],
    chat_history: list[dict[str, str]] | None = None,
    *,
    few_shot_examples: list[dict[str, str]] | None = _SENTINEL,  # type: ignore[assignment]
    system_prompt: str | None = None,
    user_template: str | None = None,
    max_context_chars: int | None = None,
) -> list[dict[str, str]]:
    """Build a structured message list for the LLM generation step.

    Parameters
    ----------
    question:
        The user's current question.
    contexts:
        Retrieved / reranked chunks to use as reference.  Each element
        can be a plain ``str`` or a :class:`~src.retrieval.reranker.RerankerResult`.
    chat_history:
        Prior turns as ``[{"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}, ...]``.  Placed
        between the system message and the current user message.
    few_shot_examples:
        List of ``{"role": ..., "content": ...}`` dicts to inject into
        the system prompt.  Pass ``[]`` to disable.  Defaults to
        :data:`LASER_FEW_SHOT_EXAMPLES`.
    system_prompt:
        Override the system prompt template.  Must contain
        ``{few_shot_examples}`` if few-shot injection is desired.
        Default from ``RAG_SYSTEM_PROMPT`` config.
    user_template:
        Override the user-message template.  Must contain
        ``{context}`` and ``{question}``.  Default from
        ``RAG_PROMPT_TEMPLATE`` config.
    max_context_chars:
        Truncate the formatted context block to this many characters
        (after individual chunk truncation).  Default from
        ``GENERATION_MAX_CONTEXT_CHARS`` config.

    Returns
    -------
    list[dict[str, str]]
        Messages ready for ``client.chat.completions.create(messages=...)``.
    """
    from config import settings

    _s: Settings = settings

    # ---- resolve defaults --------------------------------------------
    if system_prompt is None:
        system_prompt = _s.rag_system_prompt
    if user_template is None:
        user_template = _s.rag_prompt_template
    if max_context_chars is None:
        max_context_chars = _s.generation_max_context_chars
    if few_shot_examples is _SENTINEL:
        few_shot_examples = LASER_FEW_SHOT_EXAMPLES

    chat_history = chat_history or []

    # ---- format contexts ---------------------------------------------
    context_text = _format_contexts(contexts, max_context_chars)

    # ---- build system message ----------------------------------------
    fs_text = _format_few_shots(few_shot_examples)
    system_content = system_prompt.format(few_shot_examples=fs_text)

    # ---- build user message ------------------------------------------
    user_content = user_template.format(
        context=context_text,
        question=question,
    )

    # ---- assemble ----------------------------------------------------
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
    ]
    # validate & inject chat history
    for msg in chat_history:
        role = msg.get("role", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": msg.get("content", "")})

    messages.append({"role": "user", "content": user_content})

    logger.info(
        "Prompt built: system=%d chars, history=%d turns, user=%d chars",
        len(system_content),
        len(chat_history) // 2,
        len(user_content),
    )
    return messages


# ===========================================================================
#  internal helpers
# ===========================================================================

def _format_contexts(
    contexts: Sequence[Union[str, RerankerResult]],
    max_chars: int,
) -> str:
    """Format context chunks into a single string with source labels.

    Each chunk is prefixed with its source filename and page (when
    available).  The whole block is truncated to *max_chars* using a
    fair-truncation strategy: each chunk gets at most ``max_chars / N``
    characters before truncation.
    """
    if not contexts:
        return "（暂无参考文档）"

    parts: list[str] = []
    n = len(contexts)

    # per-chunk budget for fair distribution
    per_chunk = max(300, max_chars // n) if n > 0 else max_chars

    for i, ctx in enumerate(contexts):
        if isinstance(ctx, str):
            text = ctx
            source = "未知来源"
            page = None
        else:
            text = ctx.text
            source = ctx.filename or "未知来源"
            page = ctx.page

        # source label — for internal reference only; the model is
        # instructed NOT to repeat these markers in its output
        label = f"「{source}」"
        if page is not None:
            label = f"「{source}，第{page}页」"

        # truncate individual chunk if needed
        if len(text) > per_chunk:
            text = text[:per_chunk] + "…"

        parts.append(f"--- {label} ---\n{text}")

    # join and apply global truncation
    full = "\n\n".join(parts)

    if len(full) > max_chars:
        # cut at nearest newline boundary before the limit
        cut = full.rfind("\n", 0, max_chars)
        if cut > max_chars // 2:
            full = full[:cut] + "\n…（上下文过长，已截断）"

    return full


def _format_few_shots(examples: list[dict[str, str]] | None) -> str:
    """Render few-shot examples into a string for system-prompt injection.

    Returns an empty string when *examples* is ``None`` or empty so
    that ``"{few_shot_examples}"`` is a no-op in the system template.
    """
    if not examples:
        return ""

    lines: list[str] = [
        "",
        "## 示例（请严格遵循以下回答风格）：",
        "",
    ]
    for ex in examples:
        role = ex.get("role", "")
        content = ex.get("content", "")
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            lines.append(f"助手：{content}")
        lines.append("")

    return "\n".join(lines)
