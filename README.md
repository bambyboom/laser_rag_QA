<h1 align="center">⚡ Laser RAG QA</h1>
<p align="center"><strong>激光器行业知识库问答系统</strong></p>
<p align="center">基于 RAG（检索增强生成）的专业激光器知识问答，覆盖固体激光器、气体激光器、半导体激光器、光纤激光器等领域。</p>

---

## 架构概览

```
用户问题
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  QueryRewriter      — 多查询改写（提升召回率）         │
│  HybridRetriever    — 向量 (BGE-M3) + BM25 → RRF 融合 │
│  Reranker           — LLM 打分 1~10 → Top-N 精选      │
│  build_prompt()     — 结构化 System/User Message      │
│  Generator          — SSE 流式生成 + 引用提取          │
└─────────────────────────────────────────────────────┘
  │
  ▼
FastAPI /api/v1/chat  →  SSE (text/event-stream)
  │
  ▼
单页前端  →  实时流式渲染 + 引用折叠 + 错误重试
```

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+
python3 --version

# 克隆项目（或直接进入目录）
cd laser_rag_QA

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 从模板创建配置文件
cp .env.example .env

# 编辑 .env，填入 API 密钥
#   EMBEDDING_API_KEY=sk-xxxxxxxx    ← BGE-M3 嵌入服务密钥
#   LLM_API_KEY=sk-xxxxxxxx          ← DeepSeek / OpenAI 密钥
vim .env
```

### 3. 准备文档

```bash
# 将激光器技术文档放入 data/raw/ 目录
# 支持格式：PDF、DOCX、TXT、Markdown、HTML
cp ~/your-laser-docs/*.pdf data/raw/
```

### 4. 初始化数据库

```bash
# 检查集合状态
python scripts/init_db.py --stats

# （可选）重置数据库
python scripts/init_db.py --reset
```

### 5. 构建知识库

```bash
# 批量导入 data/raw/ 下的所有文档
python scripts/ingest_docs.py

# 输出示例：
#   20:15:01 | INFO    | laser-rag.scripts.ingest | Ingest complete — 384 chunk(s) now in vector store
```

### 6. 启动服务

```bash
python main.py

# 访问 http://localhost:8000
# API 文档：http://localhost:8000/docs
```

## 配置参考

所有配置通过 `.env` 文件或环境变量管理，完整列表见 [config.py](config.py) 和 [.env.example](.env.example)。

| 分类 | 变量 | 默认值 | 说明 |
|---|---|---|---|
| **嵌入** | `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | 嵌入模型 |
| | `EMBEDDING_API_KEY` | — | **必填**，嵌入 API 密钥 |
| | `EMBEDDING_API_BASE` | `https://api.openai.com/v1` | 嵌入 API 地址 |
| **LLM** | `LLM_MODEL_NAME` | `deepseek-v4-pro` | 改写/重排模型 |
| | `LLM_API_KEY` | — | **必填**，LLM API 密钥 |
| | `LLM_API_BASE` | `https://api.deepseek.com/v1` | LLM API 地址 |
| | `LLM_TEMPERATURE` | `0.1` | 生成温度 |
| | `LLM_MAX_TOKENS` | `2048` | 最大输出 token |
| **生成** | `GENERATOR_MODEL_NAME` | `gpt-4o` | 回答生成模型（可独立配置） |
| **向量库** | `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB 持久化目录 |
| | `CHROMA_COLLECTION_NAME` | `laser_knowledge` | 集合名称 |
| **检索** | `RETRIEVAL_TOP_K` | `5` | 向量检索 Top-K |
| | `RETRIEVAL_HYBRID_RRF_K` | `60` | RRF 融合参数 |
| **重排** | `RERANKER_ENABLED` | `true` | 是否启用重排 |
| | `RERANKER_TOP_N` | `5` | 重排后保留数 |
| **分块** | `CHUNK_SIZE` | `512` | 文本块大小 |
| | `CHUNK_OVERLAP` | `64` | 块间重叠 |
| **服务** | `SERVER_HOST` | `0.0.0.0` | 监听地址 |
| | `SERVER_PORT` | `8000` | 监听端口 |
| | `CORS_ORIGINS` | `*` | 允许的跨域来源 |

## API 端点

### POST `/api/v1/chat` — RAG 问答 (SSE 流式)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Nd:YAG激光器的典型输出功率范围是多少？"}' \
  --no-buffer
```

SSE 事件流：

```
event: token
data: {"content":"Nd:YAG"}

event: token
data: {"content":"激光器的典型输出功率为10-500W[1]。"}

event: references
data: {"references":[{"index":1,"source":"Nd:YAG手册.pdf","text":"...","page":3}]}

event: done
data: {"finish_reason":"stop","total_tokens":156}
```

<details>
<summary>带历史的请求</summary>

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

### POST `/api/v1/ingest/file` — 上传文档

```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "files=@laser_manual.pdf" \
  -F "files=@spec_sheet.docx"
```

### POST `/api/v1/ingest/text` — 直接入库文本

```bash
curl -X POST http://localhost:8000/api/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "YAG激光器的波长为1064nm...", "filename": "yag_basic.txt"}'
```

### POST `/api/v1/ingest/run` — 批量入库

```bash
curl -X POST http://localhost:8000/api/v1/ingest/run
```

### GET `/api/v1/health` — 健康检查

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","version":"0.1.0","index_size":384}
```

## CLI 工具

```bash
# 初始化 / 管理向量库
python scripts/init_db.py --stats          # 查看统计
python scripts/init_db.py --reset          # 重置集合
python scripts/init_db.py --clear          # 清空文档
python scripts/init_db.py --reset -y       # 跳过确认

# 批量导入文档
python scripts/ingest_docs.py              # 导入 data/raw/
python scripts/ingest_docs.py --dir /path  # 指定目录
python scripts/ingest_docs.py --reset      # 先清空再导入
python scripts/ingest_docs.py -v           # 详细日志
```

## 项目结构

```
laser_rag_QA/
├── main.py                      # FastAPI 入口 + 静态文件挂载
├── config.py                    # 全局配置（.env → Settings dataclass）
├── requirements.txt             # Python 依赖
├── .env.example                 # 配置模板
│
├── frontend/                    # 单页前端
│   ├── index.html               # 聊天 UI 结构
│   ├── style.css                # 暗色科技风主题
│   └── app.js                   # SSE 流式消费 + 引用展示 + 重试
│
├── src/
│   ├── api/
│   │   ├── routes.py            # FastAPI 路由 (/chat, /ingest/*, /health)
│   │   └── schemas.py           # Pydantic 请求/响应模型
│   │
│   ├── pipeline/
│   │   ├── ingest.py            # 文档入库流水线
│   │   └── qa.py                # RAG 问答流水线 (改写→检索→重排→生成)
│   │
│   ├── retrieval/
│   │   ├── query_rewriter.py    # 查询多路改写
│   │   ├── retriever.py         # 混合检索 (向量 + BM25 → RRF)
│   │   ├── reranker.py          # LLM 重排序
│   │   └── bm25.py              # BM25 关键词检索 (jieba 分词)
│   │
│   ├── generation/
│   │   ├── generator.py         # LLM 生成器 (SSE 流式 + 引用提取)
│   │   └── prompt.py            # System/User Prompt 构建
│   │
│   ├── knowledge/
│   │   ├── loader.py            # 多格式文档加载器
│   │   ├── cleaner.py           # 文本清洗
│   │   ├── splitter.py          # 智能分块 (512 token, 64 overlap)
│   │   ├── embedder.py          # BGE-M3 嵌入 (OpenAI API)
│   │   └── store.py             # ChromaDB 向量库封装
│   │
│   └── processing/              # (预留) 后处理 / 缓存
│
├── scripts/
│   ├── init_db.py               # 数据库初始化 & 管理
│   └── ingest_docs.py           # 批量导入脚本
│
├── data/
│   └── raw/                     # 原始文档存放目录
│
└── chroma_db/                   # ChromaDB 持久化数据 (自动生成)
```

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| **Web 框架** | FastAPI + Uvicorn | 异步 HTTP，原生 SSE 支持 |
| **向量库** | ChromaDB | 持久化，HNSW 索引，余弦相似度 |
| **嵌入模型** | BGE-M3 (1024d) | 中英文双语，OpenAI 兼容 API |
| **LLM** | DeepSeek v4 Pro | 生成 / 改写 / 重排 |
| **关键词检索** | BM25 + jieba | 中文分词 |
| **融合算法** | RRF (k=60) | 向量 + BM25 结果融合 |
| **前端** | Vanilla HTML/CSS/JS | 零依赖，fetch SSE 流式消费 |
| **文档解析** | PyPDF + python-docx + BS4 | PDF/DOCX/TXT/MD/HTML |

## License

MIT
