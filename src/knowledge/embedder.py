from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from src.knowledge.splitter import TextChunk

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger("laser-rag.embedder")

# ---------------------------------------------------------------------------
#  output structure
# ---------------------------------------------------------------------------


@dataclass
class EmbeddedChunk:
    """A chunk paired with its embedding vector, ready for ChromaDB ingestion."""

    id: str
    text: str
    embedding: list[float]
    filename: str = ""
    chunk_id: int = 0
    page: int | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Embedder — supports both local (sentence-transformers) and API modes
# ---------------------------------------------------------------------------


class Embedder:
    """Embedding client with local and API backends.

    When ``EMBEDDING_LOCAL=true``, uses ``sentence-transformers`` to load
    a BGE-M3 (or other HuggingFace) model locally.  Otherwise uses the
    OpenAI-compatible embeddings API.

    Parameters
    ----------
    model: Embedding model name (default from ``EMBEDDING_MODEL_NAME`` or
           ``EMBEDDING_LOCAL_MODEL`` config).
    api_key: API key (default from ``EMBEDDING_API_KEY`` config).  Not
             needed in local mode.
    api_base: Base URL for the embeddings endpoint (API mode only).
    batch_size: Max texts per call (API mode: per request;
                local mode: per encode() call).
    max_retries: Total attempts including the initial call (API mode only).
    timeout: Per-request timeout in seconds (API mode only).
    device: Device for local model (default from ``EMBEDDING_DEVICE``,
            e.g. ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        batch_size: int = 64,
        max_retries: int | None = None,
        timeout: float | None = None,
        device: str | None = None,
    ) -> None:
        from config import settings

        self._s: Settings = settings

        self._local = self._s.embedding_local
        self.batch_size = batch_size

        if self._local:
            # ---- local mode -----------------------------------------------
            self.model = model or self._s.embedding_local_model
            _device = device or self._s.embedding_device

            logger.info("Loading local embedding model %r on %s ...", self.model, _device)
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: pip install sentence-transformers"
                )

            self._local_model = SentenceTransformer(
                self.model,
                device=_device,
                trust_remote_code=True,
                local_files_only=True,
            )
            _dim = self._local_model.get_sentence_embedding_dimension()
            logger.info(
                "Local embedding model loaded: %s (dim=%d, device=%s)",
                self.model,
                _dim,
                _device,
            )
            self._client = None  # not used in local mode
        else:
            # ---- API mode -------------------------------------------------
            from openai import APIError, APITimeoutError, OpenAI, RateLimitError
            from tenacity import (
                before_sleep_log,
                retry,
                retry_if_exception,
                stop_after_attempt,
                wait_exponential,
            )

            self.model = model or self._s.embedding_model_name
            self._max_retries = (
                max_retries if max_retries is not None else self._s.embedding_max_retries
            )
            self._timeout = (
                timeout if timeout is not None else self._s.embedding_timeout
            )

            _key = api_key or self._s.embedding_api_key
            _base = api_base or self._s.embedding_api_base

            if not _key:
                raise ValueError(
                    "Embedding API key not set. Provide it via EMBEDDING_API_KEY "
                    "or OPENAI_API_KEY environment variable, or set EMBEDDING_LOCAL=true "
                    "to use a local model."
                )

            self._client = OpenAI(api_key=_key, base_url=_base, timeout=self._timeout)

            # bind retry helpers as instance methods for the API path
            def _is_retryable(exc: BaseException) -> bool:
                if isinstance(exc, RateLimitError):
                    return True
                if isinstance(exc, APITimeoutError):
                    return True
                if isinstance(exc, APIError):
                    status = getattr(exc, "status_code", None)
                    if status is None:
                        return True
                    return status >= 500
                if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
                    return True
                return False

            self._retry_call = retry(
                retry=retry_if_exception(_is_retryable),
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=60),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )
            self._local_model = None

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for *texts* (order-preserving).

        Texts are batched automatically.  Each inner list has
        dimension matching the model output.
        """
        if not texts:
            return []

        if self._local:
            return self._embed_local(texts)
        else:
            return self._embed_api(texts)

    def embed_chunks(self, chunks: Sequence[TextChunk]) -> list[EmbeddedChunk]:
        """Embed a sequence of TextChunks, returning EmbeddedChunk objects."""
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        vectors = self.embed(texts)

        results: list[EmbeddedChunk] = []
        for chunk, vector in zip(chunks, vectors):
            chunk_id = f"{chunk.filename}:{chunk.chunk_id}"
            results.append(
                EmbeddedChunk(
                    id=chunk_id,
                    text=chunk.text,
                    embedding=vector,
                    filename=chunk.filename,
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    metadata=dict(chunk.metadata),
                )
            )

        logger.info(
            "Embedded %d chunks → %d vectors (dim=%d)",
            len(chunks),
            len(results),
            len(vectors[0]) if vectors else 0,
        )
        return results

    # ------------------------------------------------------------------
    #  local backend
    # ------------------------------------------------------------------
    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Encode texts with a local sentence-transformers model."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialised")

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug(
                "Local embed batch %d/%d (%d texts)",
                i // self.batch_size + 1,
                (len(texts) + self.batch_size - 1) // self.batch_size,
                len(batch),
            )
            # sentence-transformers encode returns a numpy array
            vectors = self._local_model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(vectors.tolist())

        return all_embeddings

    # ------------------------------------------------------------------
    #  API backend
    # ------------------------------------------------------------------
    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """Encode texts via the OpenAI-compatible embeddings API with retry."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug(
                "API embed batch %d/%d (%d texts)",
                i // self.batch_size + 1,
                (len(texts) + self.batch_size - 1) // self.batch_size,
                len(batch),
            )

            @self._retry_call
            def _do_call():
                return self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                )

            response = _do_call()
            # normalize by input order (API returns results in input order)
            sorted_data = sorted(response.data, key=lambda d: d.index)
            batch_vectors = [d.embedding for d in sorted_data]
            all_embeddings.extend(batch_vectors)

        return all_embeddings
