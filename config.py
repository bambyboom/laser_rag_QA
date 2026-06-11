from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# load .env (if present) — must happen before any os.getenv calls
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# project root (directory containing this config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# helper: read env with type coercion
# ---------------------------------------------------------------------------
def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(key: str, default: str) -> Path:
    return Path(_env(key, default))


# ===========================================================================
#  Embedding  (BGE-M3 via OpenAI-compatible API or local model)
# ===========================================================================
EMBEDDING_LOCAL: bool = _env_bool("EMBEDDING_LOCAL", False)
EMBEDDING_LOCAL_MODEL: str = _env("EMBEDDING_LOCAL_MODEL", "BAAI/bge-m3")
EMBEDDING_MODEL_NAME: str = _env("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_API_KEY: str = _env("EMBEDDING_API_KEY", _env("OPENAI_API_KEY", ""))
EMBEDDING_API_BASE: str = _env("EMBEDDING_API_BASE", "https://api.openai.com/v1")
EMBEDDING_DIMENSIONS: int = _env_int("EMBEDDING_DIMENSIONS", 1024)
EMBEDDING_MAX_RETRIES: int = _env_int("EMBEDDING_MAX_RETRIES", 3)
EMBEDDING_TIMEOUT: float = _env_float("EMBEDDING_TIMEOUT", 60.0)
EMBEDDING_DEVICE: str = _env("EMBEDDING_DEVICE", "cpu")

# ===========================================================================
#  LLM / Generation  (DeepSeek v4 Pro)
# ===========================================================================
LLM_MODEL_NAME: str = _env("LLM_MODEL_NAME", "deepseek-v4-pro")
LLM_API_KEY: str = _env("LLM_API_KEY", _env("OPENAI_API_KEY", ""))
LLM_API_BASE: str = _env("LLM_API_BASE", "https://api.deepseek.com/v1")
LLM_MAX_TOKENS: int = _env_int("LLM_MAX_TOKENS", 2048)
LLM_TEMPERATURE: float = _env_float("LLM_TEMPERATURE", 0.1)
LLM_TIMEOUT: float = _env_float("LLM_TIMEOUT", 120.0)
LLM_MAX_RETRIES: int = _env_int("LLM_MAX_RETRIES", 2)

# ===========================================================================
#  Generator  (answer generation & streaming)
# ===========================================================================
GENERATOR_MODEL_NAME: str = _env("GENERATOR_MODEL_NAME", "gpt-4o")

# ===========================================================================
#  ChromaDB
# ===========================================================================
CHROMA_PERSIST_DIR: Path = _env_path("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION_NAME: str = _env("CHROMA_COLLECTION_NAME", "laser_knowledge")

# ===========================================================================
#  Data & Documents
# ===========================================================================
DATA_DIR: Path = _env_path("DATA_DIR", str(PROJECT_ROOT / "data"))
SUPPORTED_SUFFIXES: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx", ".html")

# ===========================================================================
#  Text cleaning
# ===========================================================================
CLEANER_NOISE_PATTERNS: list[str] = _env(
    "CLEANER_NOISE_PATTERNS",
    "激光器选型手册,版本号,修订记录,第,页,Page,Revision,Confidential",
).split(",")
CLEANER_MIN_LINE_LENGTH: int = _env_int("CLEANER_MIN_LINE_LENGTH", 2)
CLEANER_MAX_CONSECUTIVE_BLANKS: int = _env_int("CLEANER_MAX_CONSECUTIVE_BLANKS", 1)
CLEANER_REMOVE_REPEATED_LINES: bool = _env_bool("CLEANER_REMOVE_REPEATED_LINES", True)
CLEANER_REPEATED_LINE_RATIO: float = _env_float("CLEANER_REPEATED_LINE_RATIO", 0.3)

# ===========================================================================
#  Text chunking
# ===========================================================================
CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 512)
CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 64)
CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", "。", ".", " ", ""]

# ===========================================================================
#  Retrieval
# ===========================================================================
RETRIEVAL_TOP_K: int = _env_int("RETRIEVAL_TOP_K", 5)
RETRIEVAL_SIMILARITY_THRESHOLD: float = _env_float("RETRIEVAL_SIMILARITY_THRESHOLD", 0.0)
RETRIEVAL_MAX_CHUNK_TOKENS: int = _env_int("RETRIEVAL_MAX_CHUNK_TOKENS", 4096)
RETRIEVAL_HYBRID_RRF_K: int = _env_int("RETRIEVAL_HYBRID_RRF_K", 60)
RETRIEVAL_HYBRID_EXPAND: int = _env_int("RETRIEVAL_HYBRID_EXPAND", 3)

# ===========================================================================
#  Reranker
# ===========================================================================
RERANKER_ENABLED: bool = _env_bool("RERANKER_ENABLED", True)
RERANKER_TOP_N: int = _env_int("RERANKER_TOP_N", 5)
RERANKER_MAX_CHUNK_CHARS: int = _env_int("RERANKER_MAX_CHUNK_CHARS", 800)

# ===========================================================================
#  Web Search  (DuckDuckGo)
# ===========================================================================
WEB_SEARCH_ENABLED: bool = _env_bool("WEB_SEARCH_ENABLED", True)
WEB_SEARCH_MAX_RESULTS: int = _env_int("WEB_SEARCH_MAX_RESULTS", 5)
WEB_SEARCH_REGION: str = _env("WEB_SEARCH_REGION", "wt-wt")
WEB_SEARCH_TIMELIMIT: str | None = _env("WEB_SEARCH_TIMELIMIT", "") or None

# ===========================================================================
#  Routing
# ===========================================================================
RERANK_KB_SUFFICIENT_THRESHOLD: int = _env_int("RERANK_KB_SUFFICIENT_THRESHOLD", 7)
RERANK_RELEVANT_CHUNK_MIN: int = _env_int("RERANK_RELEVANT_CHUNK_MIN", 4)
RERANK_RELEVANT_CHUNK_COUNT: int = _env_int("RERANK_RELEVANT_CHUNK_COUNT", 2)
RERANK_OUT_OF_DOMAIN_THRESHOLD: int = _env_int("RERANK_OUT_OF_DOMAIN_THRESHOLD", 3)

# ===========================================================================
#  OCR — Scanned PDF to text via Tesseract
# ===========================================================================
OCR_ENABLED: bool = _env_bool("OCR_ENABLED", True)
OCR_LANGUAGE: str = _env("OCR_LANGUAGE", "chi_sim+eng")
OCR_DPI: int = _env_int("OCR_DPI", 300)
OCR_FORCE: bool = _env_bool("OCR_FORCE", False)
OCR_MIN_TEXT_LENGTH: int = _env_int("OCR_MIN_TEXT_LENGTH", 50)
OCR_TESSERACT_CMD: str = _env("OCR_TESSERACT_CMD", "tesseract")
OCR_TESSDATA_PREFIX: str = _env("OCR_TESSDATA_PREFIX", "")

# ===========================================================================
#  Prompt template
# ===========================================================================
RAG_SYSTEM_PROMPT: str = _env(
    "RAG_SYSTEM_PROMPT",
    (
        "你是激光器行业知识库的顶级专家助手，兼具深厚的理论基础与工程实践经验。"
        "你的核心使命：基于知识库文档提供精准答案；当文档信息不足时，主动调用"
        "广域知识进行补充，并明确标注来源性质。\n\n"
        "## 回答框架\n"
        "1. **答案概览**：用 1–3 句话给出核心结论\n"
        "2. **分点详述**：按主题维度展开，每点聚焦一个独立信息单元\n"
        "3. **来源参考**：在回答末尾列出引用的知识库文档清单（如有）\n\n"
        "## 去重与整合规则（极其重要）\n"
        "参考文档中的多个片段可能存在高度重叠的内容。在回答前执行以下信息整合：\n"
        "- **语义归并**：将不同片段中描述同一事实、同一参数、同一结论的内容合并"
        "为一个信息单元，只陈述一次，绝不复述多个片段中的相同或高度相似的句子\n"
        "- **互补补全**：如果片段 A 提到参数 X 的上限，片段 B 提到参数 X 的下限，"
        "应整合为「范围为 a–b」而不是分别罗列\n"
        "- **矛盾标注**：当两个片段给出冲突信息时，明确标注矛盾并优先采纳时效"
        "性更强或来源更权威的数据\n"
        "- **去噪省略**：舍弃片段中的引言套话、章节标题残留、表格格式碎片等非"
        "实质性内容\n\n"
        "## 来源标注规则\n"
        "- **禁止在正文中使用内联来源标记**，不要出现「【来源：xxx】」「[1]」"
        "「（见xxx文档）」等行内引用\n"
        "- 所有来源引用统一放在回答末尾的「📚 参考文档」小节，使用以下格式：\n"
        "  **知识库来源**（📚 标记）：\n"
        "  > 1. 《光纤激光器原理》 — 第3页\n"
        "  > 2. 《Nd:YAG技术手册》 — 第12–14页\n"
        "  **网络补充来源**（🌐 标记，必须附带网址）：\n"
        "  > 3. 🌐 《Wikipedia - Fiber laser》 — https://en.wikipedia.org/wiki/Fiber_laser\n"
        "  > 4. 🌐 《RP Photonics Encyclopedia》 — https://www.rp-photonics.com/...\n"
        "- 如果某条信息来自多份文档的共同支持，在参考清单中一并列出，回答正文"
        "中只陈述整合后的结论\n"
        "- 🌐 网络来源必须给出完整URL，优先引用 Wikipedia、官方标准文档、学术"
        "综述等权威来源。模型提供的URL来自训练数据，可能已失效，如有条件请"
        "通过联网搜索核实\n\n"
        "## 信息不足时的处理策略\n"
        "当参考文档明显不足以完整回答用户问题时，采用以下分级策略：\n\n"
        "**第一级 — 知识库有部分信息**：先用知识库内容回答已有部分，然后明确"
        "指出缺口，并启动补充模式。\n\n"
        "**第二级 — 网络补充模式**：缺口部分使用广域知识补充，但必须：\n"
        "- 在补充内容前插入「🌐 网络补充」标签，与知识库内容形成明确视觉区隔\n"
        "- 补充内容同样遵循去重规则和结构化要求\n"
        "- 必须附带参考网址：每条网络补充的关键事实至少给出一个权威URL（如 "
        "Wikipedia、官方标准文档、学术综述等）\n"
        "- 在回答末尾注明：「⚠️ 以上标注『🌐 网络补充』的内容来自广域知识，"
        "非本地知识库文档，URL 来自模型训练数据可能已失效，请核实后再用于"
        "关键决策」\n\n"
        "**第三级 — 完全无匹配**：当知识库与问题完全不相关时，全面启用网络"
        "补充模式，以广域知识构建完整回答，并在开头声明：「⚠️ 知识库中未找到"
        "相关文档，以下回答来自广域知识（🌐 网络补充）：」。全文每条关键事实"
        "必须附带权威参考网址，格式为 🌐《来源名》— URL\n\n"
        "## 风格与质量要求\n"
        "- 结构化输出，优先使用 Markdown 标题 / 列表 / 表格\n"
        "- 激光器领域术语必须准确，参数值带单位，重要参数使用粗体\n"
        "- 回答简洁，避免冗余铺垫，直击要点\n"
        "- 当知识库信息足以完整回答时，不需要触发「网络补充」\n"
        "{few_shot_examples}"
    ),
)
RAG_PROMPT_TEMPLATE: str = _env(
    "RAG_PROMPT_TEMPLATE",
    (
        "你是激光器行业知识库专家助手。请根据以下参考资料和你的专业知识，"
        "为用户提供准确、整合后的答案。\n\n"
        "参考资料（从知识库检索到的相关片段，注意去重整合）：\n"
        "{context}\n\n"
        "用户问题：{question}\n\n"
        "请遵循系统指令中的所有规则（去重整合、禁止内联标记、信息不足时分级补充）"
        "来组织你的完整回答。"
    ),
)
GENERATION_MAX_CONTEXT_CHARS: int = _env_int("GENERATION_MAX_CONTEXT_CHARS", 6000)

# ===========================================================================
#  Server
# ===========================================================================
SERVER_HOST: str = _env("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = _env_int("SERVER_PORT", 8000)
SERVER_RELOAD: bool = _env_bool("SERVER_RELOAD", False)
CORS_ORIGINS: list[str] = _env("CORS_ORIGINS", "*").split(",")

# ===========================================================================
#  Logging
# ===========================================================================
LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
LOG_FORMAT: str = _env(
    "LOG_FORMAT",
    "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s",
)

# ===========================================================================
#  Convenience: single Settings namespace
# ===========================================================================


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all configuration values."""

    # Embedding
    embedding_local: bool = EMBEDDING_LOCAL
    embedding_local_model: str = EMBEDDING_LOCAL_MODEL
    embedding_model_name: str = EMBEDDING_MODEL_NAME
    embedding_api_key: str = EMBEDDING_API_KEY
    embedding_api_base: str = EMBEDDING_API_BASE
    embedding_dimensions: int = EMBEDDING_DIMENSIONS
    embedding_max_retries: int = EMBEDDING_MAX_RETRIES
    embedding_timeout: float = EMBEDDING_TIMEOUT
    embedding_device: str = EMBEDDING_DEVICE

    # LLM
    llm_model_name: str = LLM_MODEL_NAME
    llm_api_key: str = LLM_API_KEY
    llm_api_base: str = LLM_API_BASE
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_temperature: float = LLM_TEMPERATURE
    llm_timeout: float = LLM_TIMEOUT
    llm_max_retries: int = LLM_MAX_RETRIES

    # Generator
    generator_model_name: str = GENERATOR_MODEL_NAME

    # ChromaDB
    chroma_persist_dir: Path = CHROMA_PERSIST_DIR
    chroma_collection_name: str = CHROMA_COLLECTION_NAME

    # Data
    data_dir: Path = DATA_DIR
    supported_suffixes: tuple[str, ...] = SUPPORTED_SUFFIXES

    # OCR
    ocr_enabled: bool = OCR_ENABLED
    ocr_language: str = OCR_LANGUAGE
    ocr_dpi: int = OCR_DPI
    ocr_force: bool = OCR_FORCE
    ocr_min_text_length: int = OCR_MIN_TEXT_LENGTH
    ocr_tesseract_cmd: str = OCR_TESSERACT_CMD
    ocr_tessdata_prefix: str = OCR_TESSDATA_PREFIX

    # Cleaning
    cleaner_noise_patterns: list[str] = field(default_factory=lambda: list(CLEANER_NOISE_PATTERNS))
    cleaner_min_line_length: int = CLEANER_MIN_LINE_LENGTH
    cleaner_max_consecutive_blanks: int = CLEANER_MAX_CONSECUTIVE_BLANKS
    cleaner_remove_repeated_lines: bool = CLEANER_REMOVE_REPEATED_LINES
    cleaner_repeated_line_ratio: float = CLEANER_REPEATED_LINE_RATIO

    # Chunking
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    chunk_separators: list[str] = field(default_factory=lambda: list(CHUNK_SEPARATORS))

    # Retrieval
    retrieval_top_k: int = RETRIEVAL_TOP_K
    retrieval_similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD
    retrieval_max_chunk_tokens: int = RETRIEVAL_MAX_CHUNK_TOKENS
    retrieval_hybrid_rrf_k: int = RETRIEVAL_HYBRID_RRF_K
    retrieval_hybrid_expand: int = RETRIEVAL_HYBRID_EXPAND

    # Reranker
    reranker_enabled: bool = RERANKER_ENABLED
    reranker_top_n: int = RERANKER_TOP_N
    reranker_max_chunk_chars: int = RERANKER_MAX_CHUNK_CHARS

    # Web Search
    web_search_enabled: bool = WEB_SEARCH_ENABLED
    web_search_max_results: int = WEB_SEARCH_MAX_RESULTS
    web_search_region: str = WEB_SEARCH_REGION
    web_search_timelimit: str | None = WEB_SEARCH_TIMELIMIT

    # Routing
    rerank_kb_sufficient_threshold: int = RERANK_KB_SUFFICIENT_THRESHOLD
    rerank_relevant_chunk_min: int = RERANK_RELEVANT_CHUNK_MIN
    rerank_relevant_chunk_count: int = RERANK_RELEVANT_CHUNK_COUNT
    rerank_out_of_domain_threshold: int = RERANK_OUT_OF_DOMAIN_THRESHOLD

    # Prompt
    rag_system_prompt: str = RAG_SYSTEM_PROMPT
    rag_prompt_template: str = RAG_PROMPT_TEMPLATE
    generation_max_context_chars: int = GENERATION_MAX_CONTEXT_CHARS

    # Server
    server_host: str = SERVER_HOST
    server_port: int = SERVER_PORT
    server_reload: bool = SERVER_RELOAD
    cors_origins: list[str] = field(default_factory=lambda: list(CORS_ORIGINS))

    # Logging
    log_level: str = LOG_LEVEL
    log_format: str = LOG_FORMAT


# singleton — import this everywhere
settings = Settings()
