"""Pydantic request / response models for the Laser RAG QA API.

All public API contracts live here so that both the route handlers and
any client code can share a single source of truth.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ===========================================================================
#  Chat
# ===========================================================================


class ChatRequest(BaseModel):
    """Incoming chat request — the user's question with optional history.

    Example::

        {
            "question": "YAG激光器的典型输出功率是多少？",
            "history": [
                {"role": "user", "content": "什么是固体激光器？"},
                {"role": "assistant", "content": "固体激光器是以固体材料为工作物质的激光器..."}
            ]
        }
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user's current question.",
        examples=["YAG激光器的典型输出功率是多少？"],
    )
    history: list[dict[str, str]] | None = Field(
        default=None,
        description="Prior conversation turns as [{'role': 'user'/'assistant', 'content': '...'}, ...].",
        examples=[
            [
                {"role": "user", "content": "什么是固体激光器？"},
                {"role": "assistant", "content": "固体激光器是..."},
            ]
        ],
    )
    enable_web_search: bool = Field(
        default=False,
        description="Whether to enable web search mode for this question. "
        "When True, forces the pipeline to search the web for supplementary information.",
    )


class ReferenceModel(BaseModel):
    """A single citation reference extracted from the answer."""

    index: int = Field(..., description="1-based citation number (e.g. 1 in [1]).")
    source: str = Field(..., description="Source filename or identifier.")
    text: str = Field(..., description="Snippet of the referenced context (first 200 chars).")
    page: int | None = Field(default=None, description="Page number if available.")
    url: str | None = Field(default=None, description="URL for web-sourced references.")
    is_web: bool = Field(default=False, description="Whether this is a web source (🌐) vs local KB (📚).")


class ChatResponse(BaseModel):
    """Non-streaming chat response (used when ``stream=false`` or for debugging).

    The streaming endpoint returns SSE events instead — this model
    represents the logical content of a complete response.
    """

    answer: str = Field(..., description="The generated answer text.")
    references: list[ReferenceModel] = Field(
        default_factory=list,
        description="Citations extracted from the answer.",
    )
    finish_reason: str = Field(
        default="stop",
        description="Reason the generation stopped (stop, length, error).",
    )


# ===========================================================================
#  Ingest
# ===========================================================================


class IngestTextRequest(BaseModel):
    """Raw text ingestion payload."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw text content to ingest.",
    )
    filename: str = Field(
        default="api_text.txt",
        description="Logical filename for provenance tracking.",
    )


class IngestResponse(BaseModel):
    """Response for a single-item ingest (text or single file)."""

    status: str = Field(default="ok", description="'ok' or 'error'.")
    filename: str = Field(..., description="The source filename.")
    chunks_stored: int = Field(default=0, description="Number of chunks added to the index.")
    message: str = Field(default="", description="Human-readable summary.")


class IngestBatchResponse(BaseModel):
    """Response for batch ingest operations."""

    status: str = Field(default="ok", description="'ok' or 'error'.")
    files_processed: int = Field(default=0, description="Number of files successfully processed.")
    total_chunks: int = Field(default=0, description="Total chunks added to the index.")
    message: str = Field(default="", description="Human-readable summary.")


# ===========================================================================
#  Health
# ===========================================================================


class HealthResponse(BaseModel):
    """Lightweight health-check response."""

    status: str = Field(default="ok", description="Service health status.")
    version: str = Field(default="0.1.0", description="API version.")
    index_size: int = Field(default=0, description="Number of chunks currently indexed.")
