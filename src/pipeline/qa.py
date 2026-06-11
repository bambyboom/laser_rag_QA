"""Full-stack RAG Q&A pipeline: rewrite → retrieve → rerank → route → generate.

Provides :class:`QAPipeline` which wires together every component —
:class:`QueryRewriter`, :class:`HybridRetriever`, :class:`Reranker`,
:class:`Router`, :class:`WebSearcher`, :func:`build_prompt`, and
:class:`Generator` — into a single call that takes a user question and
streams back SSE-formatted answer events.

The routing step (NEW) sits between reranking and generation.  It
classifies each question into one of three routes:

* **KB_ONLY**    — knowledge base has sufficient info
* **REFUSE**     — out-of-domain, return static refusal
* **WEB_SEARCH** — in-domain but KB insufficient; fall back to web search

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

from src.generation.generator import Generator, _sse_event
from src.generation.prompt import (
    REFUSE_ANSWER,
    build_prompt,
    build_refuse_messages,
    build_web_prompt,
    RAG_SYSTEM_PROMPT_KB_ONLY,
)

if TYPE_CHECKING:
    from src.pipeline.router import Router
    from src.retrieval.query_rewriter import QueryRewriter
    from src.retrieval.reranker import Reranker, RerankerResult
    from src.retrieval.retriever import HybridRetriever
    from src.search.web_searcher import WebSearcher

logger = logging.getLogger("laser-rag.pipeline")


class QAPipeline:
    """End-to-end RAG Q&A pipeline with question routing.

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
    router:
        Pre-built :class:`~src.pipeline.router.Router`.
        Created lazily when ``None``.
    web_searcher:
        Pre-built :class:`~src.search.web_searcher.WebSearcher`.
        Created lazily when ``None``.
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
        generator: Generator | None = None,
        router: Router | None = None,
        web_searcher: WebSearcher | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._rewriter = rewriter
        self._generator = generator
        self._router = router
        self._web_searcher = web_searcher

    # ------------------------------------------------------------------
    #  public API — async (for FastAPI SSE endpoints)
    # ------------------------------------------------------------------

    async def run(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        enable_web_search: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Execute the full RAG pipeline and stream SSE events.

        Steps:
        1. retrieval + reranking (sync, offloaded to thread pool)
        2. routing (async, on the event loop)
        3. route-specific handling → generation (async streaming)

        Parameters
        ----------
        question:
            The user's current question.
        history:
            Prior conversation turns, if any.
            Format: ``[{"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}, ...]``.
        enable_web_search:
            Whether the user explicitly requested web search mode.

        Yields
        ------
        str
            SSE-formatted event strings ready to stream to the client.
            See :class:`Generator` for the event reference.
        """
        loop = asyncio.get_running_loop()

        # ---- step 1: retrieval + reranking (sync, offloaded) ----
        def _retrieve_and_rerank() -> list[RerankerResult]:
            retriever = self._get_retriever()
            reranker = self._get_reranker()

            candidates = retriever.retrieve(question)
            if not candidates:
                logger.info("No candidates retrieved for %r", question[:60])
                return []
            return list(reranker.rerank(question, candidates))

        top_chunks = await loop.run_in_executor(None, _retrieve_and_rerank)

        # ---- step 2: routing (async, on the event loop) ----
        from src.pipeline.router import Route

        router = self._get_router()
        route = await router.classify(
            question, top_chunks, enable_web_search=enable_web_search
        )

        # ---- step 3: route-specific handling ----

        # --- ROUTE 2: refuse ---
        if route == Route.ROUTE_REFUSE:
            yield _sse_event("token", {"content": REFUSE_ANSWER})
            yield _sse_event("done", {"finish_reason": "stop"})
            return

        # --- ROUTE 3: web search ---
        if route == Route.ROUTE_WEB_SEARCH:
            searcher = self._get_web_searcher()
            web_response = await searcher.search(question)
            web_context_str = web_response.to_context_block()
            web_refs = web_response.to_references()

            def _build_web() -> tuple[list[dict[str, str]], list]:
                messages = build_web_prompt(
                    question,
                    top_chunks,
                    web_context_str,
                    chat_history=history,
                )
                # Combine KB + web contexts for citation extraction
                all_contexts: list = list(top_chunks) + web_refs
                return messages, all_contexts

            messages, all_contexts = await loop.run_in_executor(
                None, _build_web
            )
        else:
            # --- ROUTE 1: KB-only ---
            def _build_kb() -> tuple[list[dict[str, str]], list]:
                if not top_chunks:
                    # Edge case: no KB results but router said KB_ONLY.
                    # Fall back to empty contexts with KB-only prompt.
                    return build_prompt(
                        question, [],
                        chat_history=history,
                        system_prompt=RAG_SYSTEM_PROMPT_KB_ONLY,
                    ), []
                messages = build_prompt(
                    question, top_chunks,
                    chat_history=history,
                    system_prompt=RAG_SYSTEM_PROMPT_KB_ONLY,
                )
                return messages, list(top_chunks)

            messages, all_contexts = await loop.run_in_executor(
                None, _build_kb
            )

        # ---- step 4: generation (async streaming) ----
        async for event in self._get_generator().generate(
            messages, all_contexts
        ):
            yield event

    # ------------------------------------------------------------------
    #  public API — sync (for CLI / debugging)
    # ------------------------------------------------------------------

    def run_sync(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        enable_web_search: bool = False,
    ) -> Generator[str, None, None]:
        """Synchronous version of :meth:`run` — yields the same SSE events.

        Use this in CLI tools, scripts, or when an event loop is not
        available (e.g. a plain ``main()`` function).

        Note: domain classification and web search are inherently async
        (they use ``AsyncOpenAI`` and the thread-pool pattern).  This
        method wraps them with :func:`asyncio.run` internally.
        """
        # ---- retrieval ------------------------------------------------
        retriever = self._get_retriever()
        candidates = retriever.retrieve(question)

        top_chunks: list[RerankerResult] = []
        if candidates:
            reranker = self._get_reranker()
            top_chunks = list(reranker.rerank(question, candidates))
        else:
            logger.info("No candidates retrieved for %r", question[:60])

        # ---- routing ---------------------------------------------------
        from src.pipeline.router import Route

        async def _classify() -> Route:
            return await self._get_router().classify(
                question, top_chunks, enable_web_search=enable_web_search
            )

        route = asyncio.run(_classify())

        # ---- route-specific --------------------------------------------

        if route == Route.ROUTE_REFUSE:
            yield _sse_event("token", {"content": REFUSE_ANSWER})
            yield _sse_event("done", {"finish_reason": "stop"})
            return

        generator = self._get_generator()

        if route == Route.ROUTE_WEB_SEARCH:
            searcher = self._get_web_searcher()
            web_response = asyncio.run(searcher.search(question))
            web_context_str = web_response.to_context_block()
            web_refs = web_response.to_references()

            messages = build_web_prompt(
                question,
                top_chunks,
                web_context_str,
                chat_history=history,
            )
            all_contexts: list = list(top_chunks) + web_refs
        else:
            if not top_chunks:
                messages = build_prompt(
                    question, [],
                    chat_history=history,
                    system_prompt=RAG_SYSTEM_PROMPT_KB_ONLY,
                )
                all_contexts = []
            else:
                messages = build_prompt(
                    question, top_chunks,
                    chat_history=history,
                    system_prompt=RAG_SYSTEM_PROMPT_KB_ONLY,
                )
                all_contexts = list(top_chunks)

        yield from generator.generate_sync(messages, all_contexts)

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

    def _get_router(self) -> Router:
        if self._router is None:
            from src.pipeline.router import Router
            self._router = Router()
        return self._router

    def _get_web_searcher(self) -> WebSearcher:
        if self._web_searcher is None:
            from src.search.web_searcher import WebSearcher
            from config import settings

            self._web_searcher = WebSearcher(
                max_results=settings.web_search_max_results,
                region=settings.web_search_region,
                timelimit=settings.web_search_timelimit,
            )
        return self._web_searcher
