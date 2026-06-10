"""Full-stack RAG Q&A pipeline: rewrite → retrieve → rerank → generate.

Provides :class:`QAPipeline` which wires together every component —
:class:`QueryRewriter`, :class:`HybridRetriever`, :class:`Reranker`,
:func:`build_prompt`, and :class:`Generator` — into a single call that
takes a user question and streams back SSE-formatted answer events.

Usage::

    from src.pipeline.qa import QAPipeline

    qa = QAPipeline()

    # server-side — async streaming
    async for sse_event in qa.run("YAG激光器功率？", history=history):
        yield sse_event

    # CLI / debugging — sync streaming
    for sse_event in qa.run_sync("YAG激光器功率？"):
        print(sse_event)
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncGenerator, Generator, Sequence

from src.generation.generator import Generator
from src.generation.prompt import build_prompt

if TYPE_CHECKING:
    from src.retrieval.query_rewriter import QueryRewriter
    from src.retrieval.reranker import Reranker
    from src.retrieval.retriever import HybridRetriever

logger = logging.getLogger("laser-rag.pipeline")


class QAPipeline:
    """End-to-end RAG Q&A pipeline.

    Lazy-initialises every sub-component on first use so that imports
    stay fast and configuration is read once at call time.

    Parameters
    ----------
    retriever:
        Pre-built :class:`~src.retrieval.retriever.HybridRetriever`.
        Created lazily when ``None``.
    reranker:
        Pre-built :class:`~src.retrieval.reranker.Reranker`.
        Created lazily when ``None``.
    rewriter:
        Pre-built :class:`~src.retrieval.query_rewriter.QueryRewriter`.
        Created lazily when ``None``.
    generator:
        Pre-built :class:`Generator`.  Created lazily when ``None``.
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
        generator: Generator | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._rewriter = rewriter
        self._generator = generator

    # ------------------------------------------------------------------
    #  public API — async (for FastAPI SSE endpoints)
    # ------------------------------------------------------------------

    async def run(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Execute the full RAG pipeline and stream SSE events.

        Steps 1–3 (retrieval, reranking, prompt building) are offloaded
        to a thread-pool so the event loop stays responsive during the
        (potentially blocking) reranker LLM call.

        Parameters
        ----------
        question:
            The user's current question.
        history:
            Prior conversation turns, if any.
            Format: ``[{"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}, ...]``.

        Yields
        ------
        str
            SSE-formatted event strings ready to stream to the client.
            See :class:`Generator` for the event reference.
        """
        loop = asyncio.get_running_loop()

        # ---- steps 1–3: retrieval + reranking + prompt (sync) ----------
        def _prepare() -> tuple[list[dict[str, str]], list]:
            retriever = self._get_retriever()
            reranker = self._get_reranker()

            candidates = retriever.retrieve(question)
            if not candidates:
                logger.info("No candidates retrieved for %r", question[:60])
                return build_prompt(question, [], chat_history=history), []

            top_chunks = reranker.rerank(question, candidates)
            # Convert to plain list so we're not passing RerankerResult
            # across the executor boundary with any thread-safety issues.
            # (RerankerResult is a frozen-like dataclass, so it's fine,
            # but being explicit doesn't hurt.)
            contexts = list(top_chunks)

            messages = build_prompt(question, contexts, chat_history=history)
            return messages, contexts

        messages, contexts = await loop.run_in_executor(None, _prepare)

        # ---- step 4: generation (async streaming) -----------------------
        async for event in self._get_generator().generate(messages, contexts):
            yield event

    # ------------------------------------------------------------------
    #  public API — sync (for CLI / debugging)
    # ------------------------------------------------------------------

    def run_sync(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """Synchronous version of :meth:`run` — yields the same SSE events.

        Use this in CLI tools, scripts, or when an event loop is not
        available (e.g. a plain ``main()`` function).
        """
        retriever = self._get_retriever()
        reranker = self._get_reranker()
        generator = self._get_generator()

        # ---- retrieval -------------------------------------------------
        candidates = retriever.retrieve(question)
        if not candidates:
            logger.info("No candidates retrieved for %r", question[:60])
            messages = build_prompt(question, [], chat_history=history)
            yield from generator.generate_sync(messages, [])
            return

        # ---- reranking -------------------------------------------------
        top_chunks = reranker.rerank(question, candidates)
        contexts = list(top_chunks)

        # ---- prompt + generate ------------------------------------------
        messages = build_prompt(question, contexts, chat_history=history)
        yield from generator.generate_sync(messages, contexts)

    # ------------------------------------------------------------------
    #  health / introspection
    # ------------------------------------------------------------------

    @property
    def index_size(self) -> int:
        """Number of chunks currently indexed in the vector store."""
        try:
            return self._get_retriever().corpus_size
        except Exception:
            return 0

    # ------------------------------------------------------------------
    #  lazy initialisers
    # ------------------------------------------------------------------

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            from src.retrieval.query_rewriter import QueryRewriter
            from src.retrieval.retriever import HybridRetriever

            self._rewriter = self._rewriter or QueryRewriter()
            self._retriever = HybridRetriever(query_rewriter=self._rewriter)
        return self._retriever

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            from src.retrieval.reranker import Reranker
            self._reranker = Reranker()
        return self._reranker

    def _get_rewriter(self) -> QueryRewriter:
        if self._rewriter is None:
            from src.retrieval.query_rewriter import QueryRewriter
            self._rewriter = QueryRewriter()
        return self._rewriter

    def _get_generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator()
        return self._generator
