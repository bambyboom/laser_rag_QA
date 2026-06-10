from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings, PROJECT_ROOT  # side-effect: load_dotenv()

# ---------------------------------------------------------------------------
#  Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Laser RAG QA",
    description="激光器行业知识库问答系统",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
#  CORS (allow SSE from any origin in development)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
#  API routes (registered *before* static mount so they take priority)
# ---------------------------------------------------------------------------
from src.api.routes import router  # noqa: E402

app.include_router(router)

# ---------------------------------------------------------------------------
#  Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})

# ---------------------------------------------------------------------------
#  Static files — serves frontend/index.html at /
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "frontend"), html=True), name="static")

# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level, format=settings.log_format)
    logger = logging.getLogger("laser-rag")
    logger.info("Starting %s v%s", app.title, app.version)
    logger.info("ChromaDB persist dir : %s", settings.chroma_persist_dir)
    logger.info("LLM model           : %s", settings.llm_model_name)

    uvicorn.run(
        "main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
        log_level=settings.log_level.lower(),
    )
