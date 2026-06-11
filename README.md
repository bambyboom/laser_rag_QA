<p align="center">
  <h1 align="center">⚡ Laser RAG QA</h1>
  <p align="center">
    <strong>激光器行业知识库智能问答系统</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
    <img src="https://img.shields.io/badge/FastAPI-0.100+-teal?logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/ChromaDB-latest-orange" alt="ChromaDB">
  </p>
  <p align="center">
    基于 <strong>RAG</strong>（检索增强生成）的专业激光器知识问答系统<br>
    覆盖固体激光器 · 气体激光器 · 半导体激光器 · 光纤激光器 · 量子级联激光器等全领域
  </p>
</p>

---

## 📖 目录

- [特性](#-特性)
- [架构概览](#-架构概览)
- [快速开始](#-快速开始)
- [配置参考](#-配置参考)
- [API 文档](#-api-文档)
- [CLI 工具](#-cli-工具)
- [项目结构](#-项目结构)
- [技术栈](#-技术栈)
- [常见问题](#-常见问题)
- [License](#license)

---

## ✨ 特性

- **🔍 混合检索** — 向量检索（BGE-M3） + BM25 关键词检索，通过 RRF 算法融合，兼顾语义与关键词匹配
- **✍️ 多查询改写** — 自动将用户问题改写为多个查询变体，提升召回覆盖率
- **📊 LLM 重排序** — 对候选片段进行 1–10 精细打分，精选最相关上下文
- **🧭 智能路由** — 三路分类（知识库 / 拒答 / 联网搜索），根据重排分数 + 领域分类器自动决策
- **📡 SSE 流式生成** — 服务端推送式流式输出，支持引用标注 `[1]`、`[2]`，实时返回参考文献来源
- **📄 扫描件 OCR** — 自动检测扫描型 PDF，通过 Tesseract + PyMuPDF 渲染识别，支持中英文混排
- **🌐 联网搜索回退** — 知识库不足时自动回退到 DuckDuckGo 联网搜索（可按需开关）
- **🎨 暗色科技风 UI** — 单页纯前端，零依赖，支持 SSE 流式渲染、引用折叠、错误重试
- **📦 文档热加载** — 支持 PDF / DOCX / TXT / Markdown / HTML 多格式，API 实时上传入库

---

## 🏗 架构概览

```
用户问题 ──→ QueryRewriter ──→ HybridRetriever ──→ Reranker ──→ Router ──→ Generator ──→ SSE 流式响应
              多查询改写          向量 + BM25 → RRF   LLM 1-10 打分   三路智能路由   System Prompt   token 事件流
                                                                        │                 + 引用提取
                                                               ┌───────┼────────┐
                                                               │       │        │
                                                          KB_ONLY   REFUSE  WEB_SEARCH
                                                          知识库答   拒答     联网搜索
```

<details>
<summary><b>展开查看完整数据流</b></summary>

### 问答链路（Query-Time）

| 步骤 | 组件 | 说明 |
|:---:|------|------|
| 1 | `QueryRewriter` | 将用户问题改写为 3–4 个查询变体，覆盖不同表述角度 |
| 2 | `HybridRetriever` | 向量检索 (BGE-M3, Top-K) + BM25 关键词检索 → RRF 融合 |
| 3 | `Reranker` | LLM 对每个片段打分 1–10，精选 Top-N |
| 4 | `Router` | 根据评分 + 领域分类器，三选一：`KB_ONLY` / `REFUSE` / `WEB_SEARCH` |
| 5 | `build_prompt` | 组装 System Prompt + 参考文档 + 对话历史 |
| 6 | `Generator` | LLM 流式生成，提取引用标记，SSE 推送 |

### 入库链路（Ingest-Time）

| 步骤 | 组件 | 说明 |
|:---:|------|------|
| 1 | `DocumentLoader` | 加载 PDF / DOCX / TXT / MD / HTML |
| 1.5 | `OcrProcessor` | 自动检测扫描件 PDF → Tesseract OCR 文字识别 |
| 2 | `TextCleaner` | 去噪、去重、规范化文本 |
| 3 | `TextSplitter` | 512 token 滑动窗口分块 (64 token 重叠) |
| 4 | `Embedder` | BGE-M3 嵌入 → 1024 维向量 |
| 5 | `VectorStore` | ChromaDB 持久化存储 (HNSW 索引, 余弦相似度) |

</details>

---

## 🚀 快速开始

### 前提条件

- **Python 3.10+**
- **Tesseract OCR**（可选，仅扫描件 PDF 需要）— 见 [常见问题](#-常见问题)

### 1. 克隆项目

```bash
git clone <your-repo-url> && cd laser_rag_QA
```

### 2. 安装依赖

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
vim .env
```

**最小配置**（使用本地 BGE-M3 模型，无需 Embedding API Key）：

```ini
# 本地嵌入模式 —— 首次运行自动从 HuggingFace 下载 BGE-M3
EMBEDDING_LOCAL=true

# LLM —— 填入你的 DeepSeek API Key
LLM_API_KEY=sk-xxxxxxxx
```

> 💡 如果使用云端嵌入 API（如 SiliconFlow），设置 `EMBEDDING_LOCAL=false` 并填入 `EMBEDDING_API_KEY`。

### 4. 准备文档

```bash
# 将激光器技术文档放入 data/raw/
cp ~/your-laser-docs/*.pdf data/raw/
```

支持的格式：`.pdf` `.docx` `.txt` `.md` `.html`

### 5. 构建知识库

```bash
python scripts/ingest_docs.py

# 输出示例：
#   20:15:01 | INFO | Step 1/6  loaded 3 document(s)
#   20:15:02 | INFO | Step 1.5/6 OCR processed
#   20:15:02 | INFO | Step 2/6  cleaned → 3 document(s)
#   20:15:03 | INFO | Step 3/6  split → 384 chunk(s)
#   20:15:15 | INFO | Step 4/6  embedded → 384 vector(s) (dim=1024)
#   20:15:16 | INFO | Step 5/6  stored 384 chunk(s) in vectorstore
```

### 6. 启动服务

```bash
python main.py
```

打开浏览器访问：

| URL | 说明 |
|-----|------|
| `http://localhost:8000` | 聊天界面 |
| `http://localhost:8000/docs` | Swagger API 文档 |

---

## ⚙ 配置参考

完整的 `.env` 配置项。所有变量都有合理默认值，只需配置 **API Key** 即可运行。

### 嵌入

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_LOCAL` | `false` | `true` = 本地加载 BGE-M3（无需 API Key） |
| `EMBEDDING_LOCAL_MODEL` | `BAAI/bge-m3` | 本地模型名 / HuggingFace 路径 |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | API 模式的模型名 |
| `EMBEDDING_API_KEY` | — | 嵌入 API 密钥（本地模式不需要） |
| `EMBEDDING_API_BASE` | `https://api.openai.com/v1` | 嵌入 API 地址 |

### LLM / 生成

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL_NAME` | `deepseek-v4-pro` | 改写 / 重排 / 分类所用模型 |
| `LLM_API_KEY` | — | **必填**，LLM API 密钥 |
| `LLM_API_BASE` | `https://api.deepseek.com/v1` | LLM API 地址 |
| `LLM_TEMPERATURE` | `0.1` | 生成温度 |
| `LLM_MAX_TOKENS` | `2048` | 最大输出 token 数 |
| `LLM_TIMEOUT` | `120.0` | API 超时时间（秒） |
| `GENERATOR_MODEL_NAME` | `deepseek-v4-pro` | 回答生成模型（可独立配置） |

### 向量库

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB 持久化目录 |
| `CHROMA_COLLECTION_NAME` | `laser_knowledge` | 集合名称 |

### 检索 & 重排

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RETRIEVAL_TOP_K` | `5` | 向量检索返回数 |
| `RETRIEVAL_HYBRID_RRF_K` | `60` | RRF 融合参数 |
| `RETRIEVAL_HYBRID_EXPAND` | `3` | 混合检索扩展倍数 |
| `RERANKER_ENABLED` | `true` | 是否启用 LLM 重排 |
| `RERANKER_TOP_N` | `5` | 重排后保留数 |

### 路由

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RERANK_KB_SUFFICIENT_THRESHOLD` | `7` | Top-1 分 ≥ 此值 → KB 充足 |
| `RERANK_RELEVANT_CHUNK_MIN` | `4` | 相关片段最低分 |
| `RERANK_RELEVANT_CHUNK_COUNT` | `2` | 最少相关片段数 |
| `RERANK_OUT_OF_DOMAIN_THRESHOLD` | `3` | Top-1 分 ≤ 此值 → 可能拒答 |

### 联网搜索

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WEB_SEARCH_ENABLED` | `true` | 是否启用 DuckDuckGo 搜索 |
| `WEB_SEARCH_MAX_RESULTS` | `5` | 最大搜索结果数 |
| `WEB_SEARCH_REGION` | `wt-wt` | 搜索区域 |

### OCR（扫描件识别）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OCR_ENABLED` | `true` | 是否启用 OCR |
| `OCR_LANGUAGE` | `chi_sim+eng` | Tesseract 语言（中英文） |
| `OCR_DPI` | `300` | 页面渲染分辨率 |
| `OCR_FORCE` | `false` | `true` = 强制 OCR 所有 PDF |
| `OCR_MIN_TEXT_LENGTH` | `50` | 文本 < 此值触发 OCR |

### 分块 & 清洗

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | `512` | 文本块大小（token 数） |
| `CHUNK_OVERLAP` | `64` | 块间重叠 token 数 |
| `CLEANER_MIN_LINE_LENGTH` | `2` | 短于此值的行被丢弃 |
| `CLEANER_REMOVE_REPEATED_LINES` | `true` | 移除重复行 |

### 服务

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SERVER_HOST` | `0.0.0.0` | 监听地址 |
| `SERVER_PORT` | `8000` | 监听端口 |
| `CORS_ORIGINS` | `*` | 跨域白名单 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

> 📝 完整配置项及详细说明见 [.env.example](.env.example) 和 [config.py](config.py)。

---

## 🔌 API 文档

### `POST /api/v1/chat` — RAG 问答（SSE 流式）

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Nd:YAG激光器的典型输出功率范围是多少？"}' \
  --no-buffer
```

<details>
<summary><b>SSE 事件流格式</b></summary>

```
event: token
data: {"content":"Nd:YAG"}

event: token
data: {"content":"激光器的典型输出功率为10-500W[1]。典型电光转换效率约2%-5%[2]。"}

event: references
data: {"references":[{"index":1,"source":"Nd:YAG激光器手册.pdf","text":"...","page":3},{"index":2,"source":"固体激光工程.pdf","text":"...","page":156}]}

event: done
data: {"finish_reason":"stop","total_tokens":213}
```

</details>

<details>
<summary><b>请求体完整字段</b></summary>

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `question` | `string` | ✅ | 用户问题 |
| `history` | `list[message]` | — | 对话历史，`[{role, content}]` |
| `enable_web_search` | `bool` | — | 是否启用联网搜索（默认 `false`） |
| `temperature` | `float` | — | 覆盖默认生成温度 |
| `stream` | `bool` | — | 是否流式输出（默认 `true`） |

带历史的请求示例：

```json
{
  "question": "它的电光转换效率是多少？",
  "history": [
    {"role": "user", "content": "什么是Nd:YAG激光器？"},
    {"role": "assistant", "content": "Nd:YAG激光器是以掺钕钇铝石榴石为工作物质的固体激光器..."}
  ]
}
```

</details>

### 文档入库

```bash
# 上传文件
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "files=@laser_manual.pdf" \
  -F "files=@spec_sheet.docx"

# 直接入库文本
curl -X POST http://localhost:8000/api/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "YAG激光器的波长为1064nm...", "filename": "yag_basic.txt"}'

# 重新批量构建 data/raw/ 中的所有文档
curl -X POST http://localhost:8000/api/v1/ingest/run
```

### `GET /api/v1/health` — 健康检查

```bash
curl http://localhost:8000/api/v1/health
# → {"status":"ok","version":"0.1.0","index_size":384}
```

---

## 🛠 CLI 工具

```bash
# 数据库管理
python scripts/init_db.py --stats          # 查看向量库统计
python scripts/init_db.py --reset          # 重置集合（需确认）
python scripts/init_db.py --reset -y       # 跳过确认直接重置
python scripts/init_db.py --clear          # 清空所有文档

# 批量导入文档
python scripts/ingest_docs.py              # 导入 data/raw/
python scripts/ingest_docs.py --dir /path  # 指定目录
python scripts/ingest_docs.py --reset      # 先清空再导入
python scripts/ingest_docs.py -v           # DEBUG 详细日志
```

---

## 📁 项目结构

```
laser_rag_QA/
├── main.py                        # FastAPI 入口 + 静态文件挂载
├── config.py                      # 全局配置（.env → @dataclass Settings）
├── requirements.txt               # Python 依赖清单
├── .env.example                   # 配置模板（含注释）
│
├── frontend/                      # 单页前端（零构建）
│   ├── index.html                 # 聊天 UI 结构
│   ├── style.css                  # 暗色科技风主题
│   └── app.js                     # SSE 流式消费 + Markdown 渲染 + 引用折叠
│
├── src/
│   ├── api/
│   │   ├── routes.py              # FastAPI 路由 (/chat, /ingest, /health)
│   │   └── schemas.py             # Pydantic 请求/响应模型
│   │
│   ├── pipeline/
│   │   ├── ingest.py              # 📥 文档入库流水线（6 步）
│   │   ├── qa.py                  # 🧠 RAG 问答流水线（改写→检索→重排→路由→生成）
│   │   └── router.py              # 🧭 三路智能路由（KB / 拒答 / 联网搜索）
│   │
│   ├── retrieval/
│   │   ├── query_rewriter.py      # 多查询改写（提升召回）
│   │   ├── retriever.py           # 混合检索（BGE-M3 + BM25 → RRF）
│   │   └── reranker.py            # LLM 重排序（1-10 打分）
│   │
│   ├── generation/
│   │   ├── generator.py           # LLM 流式生成器 + 引用提取
│   │   └── prompt.py              # System / User Prompt 模板构建
│   │
│   ├── knowledge/
│   │   ├── loader.py              # 多格式文档加载器
│   │   ├── ocr.py                 # 🔍 扫描件 OCR（Tesseract + PyMuPDF）
│   │   ├── cleaner.py             # 文本去噪清洗
│   │   ├── splitter.py            # 智能分块（512 token, 64 overlap）
│   │   ├── embedder.py            # BGE-M3 嵌入（本地 / API 模式）
│   │   └── store.py               # ChromaDB 向量库封装
│   │
│   └── processing/                # （预留）后处理 / 缓存
│
├── scripts/
│   ├── init_db.py                 # 向量库初始化 & 管理
│   └── ingest_docs.py             # 批量导入 CLI
│
├── data/
│   └── raw/                       # 📂 原始文档存放目录
│
└── chroma_db/                     # ChromaDB 持久化数据（自动生成）
```

---

## 🧰 技术栈

| 层 | 技术选型 | 说明 |
|------|----------|------|
| **Web 框架** | FastAPI + Uvicorn | 异步 HTTP，原生 SSE 支持 |
| **向量库** | ChromaDB | HNSW 索引，余弦相似度，持久化 |
| **嵌入模型** | BGE-M3 (1024d) | 中英双语，支持本地 / API 两种模式 |
| **关键词检索** | BM25 | jieba 中文分词 |
| **融合算法** | RRF (k=60) | 向量 + BM25 结果融合 |
| **LLM** | DeepSeek v4 Pro | 生成 / 改写 / 重排 / 领域分类 |
| **OCR 引擎** | Tesseract 4.x | 中英文混排文字识别 |
| **PDF 渲染** | PyMuPDF (fitz) | PDF → 图片（无 poppler 依赖） |
| **前端** | Vanilla HTML/CSS/JS | 零构建，fetch SSE 流式消费 |
| **文档解析** | pypdf + python-docx | PDF / DOCX / TXT / MD / HTML |

---

## ❓ 常见问题

<details>
<summary><b>Q: 如何在没有 sudo 权限的环境安装 Tesseract？</b></summary>

使用 `.deb` 包提取到用户目录，无需 root 权限：

```bash
mkdir -p ~/.local/tesseract && cd ~/.local/tesseract
apt download tesseract-ocr tesseract-ocr-chi-sim libtesseract4 liblept5
for deb in *.deb; do dpkg-deb -x "$deb" .; done
```

然后设置环境变量：

```ini
OCR_TESSERACT_CMD=/home/your-user/.local/bin/tesseract-local
OCR_TESSDATA_PREFIX=/home/your-user/.local/tesseract/usr/share/tesseract-ocr/4.00/tessdata
```

详见 [.env.example](.env.example) 中的 OCR 配置说明。
</details>

<details>
<summary><b>Q: 本地 BGE-M3 需要联网下载吗？</b></summary>

是的，首次启动时会从 HuggingFace Hub 下载模型文件（约 2.3 GB）。如需离线使用：

```bash
# 在有网络的机器上预先下载
huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3

# 在离线环境设置
EMBEDDING_LOCAL_MODEL=./models/bge-m3
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```
</details>

<details>
<summary><b>Q: 如何处理扫描版 PDF？</b></summary>

系统会**自动检测**扫描件 PDF（pypdf 提取文本 < 50 字符或乱码比例过高）。检测到后自动调用 Tesseract OCR 进行中英文识别。无需任何额外操作，只需确保 Tesseract 已安装且 `OCR_ENABLED=true`。
</details>

<details>
<summary><b>Q: 如何强制对所有 PDF 使用 OCR？</b></summary>

设置环境变量：

```ini
OCR_FORCE=true
```

或者通过 API 参数控制。
</details>

<details>
<summary><b>Q: 联网搜索为什么不工作？</b></summary>

系统使用 DuckDuckGo 进行匿名搜索。在某些网络环境（如 WSL2、企业内网）下可能超时。系统会优雅降级到知识库内容回答。如有稳定的 Web Search API，可在配置中替换后端。
</details>

<details>
<summary><b>Q: 如何更换 LLM 模型？</b></summary>

系统使用 OpenAI 兼容 API，更换只需修改环境变量：

```ini
# 例如：切换到 SiliconFlow 的 Qwen3
LLM_MODEL_NAME=Qwen/Qwen3-235B-A22B-Instruct-2507
LLM_API_BASE=https://api.siliconflow.cn/v1
LLM_API_KEY=sk-xxxxxxxx
```
</details>

---

## License

[MIT](LICENSE)
