"""API route definitions for Laser RAG QA.

Endpoints
---------
POST /api/v1/chat          — RAG Q&A with SSE streaming
POST /api/v1/ingest/file   — Upload and ingest document files
POST /api/v1/ingest/text   — Ingest raw text
POST /api/v1/ingest/run    — Batch ingest from data/raw/
GET  /api/v1/health        — Service health check
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ChatRequest,
    HealthResponse,
    IngestBatchResponse,
    IngestResponse,
    IngestTextRequest,
)
from src.pipeline.ingest import IngestPipeline

logger = logging.getLogger("laser-rag.api")

router = APIRouter(prefix="/api/v1", tags=["laser-rag"])

# ---------------------------------------------------------------------------
#  shared pipeline instances — created once, reused across requests
# ---------------------------------------------------------------------------
_ingest_pipeline: Optional[IngestPipeline] = None
_qa_pipeline: Optional["QAPipeline"] = None  # noqa: F821


def _get_ingest_pipeline() -> IngestPipeline:
    global _ingest_pipeline
    if _ingest_pipeline is None:
        _ingest_pipeline = IngestPipeline()
    return _ingest_pipeline


def _get_qa_pipeline() -> "QAPipeline":  # noqa: F821
    global _qa_pipeline
    if _qa_pipeline is None:
        from src.pipeline.qa import QAPipeline

        _qa_pipeline = QAPipeline()
    return _qa_pipeline


# ===========================================================================
#  POST /chat  — RAG Q&A with SSE streaming
# ===========================================================================


@router.post("/chat")
async def chat(request: ChatRequest):
    """Answer a user question using the full RAG pipeline.

    Returns ``text/event-stream`` (SSE) with the following events:

    ============= ====================================================
    Event          Payload
    ============= ====================================================
    ``token``      ``{"content": "文"}`` — incremental text chunk
    ``references`` ``{"references": [...]}`` — extracted citations
    ``done``       ``{"finish_reason": "stop"}`` — completion
    ``error``      ``{"message": "..."}`` — error information
    ============= ====================================================

    Pass ``?stream=false`` to receive a single JSON response instead.
    """
    qa = _get_qa_pipeline()

    async def _event_stream():
        async for sse_chunk in qa.run(
            question=request.question,
            history=request.history,
        ):
            yield sse_chunk

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ===========================================================================
#  POST /ingest/file  — upload one or more files
# ===========================================================================


@router.post("/ingest/file", response_model=IngestBatchResponse)
async def ingest_files(
    files: list[UploadFile] = File(..., description="Documents to ingest"),
) -> IngestBatchResponse:
    """Upload and ingest one or more document files (PDF, DOCX, TXT, MD, HTML).

    Files are saved temporarily, processed through the full pipeline
    (load → clean → split → embed → store), and then cleaned up.
    """
    pipeline = _get_ingest_pipeline()
    total_chunks = 0
    processed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            if not f.filename:
                continue

            # save uploaded file to a temp location
            tmp_path = Path(tmpdir) / f.filename
            try:
                with open(tmp_path, "wb") as out:
                    shutil.copyfileobj(f.file, out)
            except Exception:
                logger.exception("Failed to save upload %s", f.filename)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save uploaded file: {f.filename}",
                )

            # process through pipeline
            try:
                chunks = pipeline.ingest_file(tmp_path)
            except Exception:
                logger.exception("Ingest failed for %s", f.filename)
                raise HTTPException(
                    status_code=500,
                    detail=f"Ingest pipeline failed for: {f.filename}",
                )

            total_chunks += chunks
            processed += 1
            logger.info("Ingested %r → %d chunk(s)", f.filename, chunks)

    return IngestBatchResponse(
        status="ok",
        files_processed=processed,
        total_chunks=total_chunks,
        message=(
            f"Successfully ingested {processed} file(s) → "
            f"{total_chunks} chunk(s)"
        ),
    )


# ===========================================================================
#  POST /ingest/text  — ingest raw text
# ===========================================================================


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(payload: IngestTextRequest) -> IngestResponse:
    """Ingest a raw text string directly (no file upload).

    Useful for programmatic or debugging scenarios.
    """
    pipeline = _get_ingest_pipeline()
    try:
        chunks = pipeline.ingest_text(
            text=payload.text,
            filename=payload.filename,
        )
    except Exception:
        logger.exception("Text ingest failed")
        raise HTTPException(status_code=500, detail="Ingest pipeline failed")

    return IngestResponse(
        status="ok",
        filename=payload.filename,
        chunks_stored=chunks,
        message=f"Ingested text as {payload.filename!r} → {chunks} chunk(s)",
    )


# ===========================================================================
#  POST /ingest/run  — trigger batch ingest from data/raw/
# ===========================================================================


@router.post("/ingest/run", response_model=IngestBatchResponse)
async def ingest_run() -> IngestBatchResponse:
    """Run the full batch ingest pipeline on ``data/raw/``.

    This is the API equivalent of ``python scripts/ingest_docs.py``.
    """
    from src.knowledge.store import VectorStore

    pipeline = _get_ingest_pipeline()
    store = VectorStore()
    before = store.count()

    try:
        total = pipeline.run()
    except Exception:
        logger.exception("Batch ingest failed")
        raise HTTPException(status_code=500, detail="Batch ingest pipeline failed")

    after = store.count()

    return IngestBatchResponse(
        status="ok",
        files_processed=after - before,
        total_chunks=total,
        message=f"Batch ingest complete — {total} chunk(s) stored",
    )


# ===========================================================================
#  GET /health  — health check
# ===========================================================================


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Lightweight health check — returns service status and index size."""
    qa = _get_qa_pipeline()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        index_size=qa.index_size,
    )
