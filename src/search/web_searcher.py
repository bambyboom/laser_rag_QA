"""Web search via DuckDuckGo, with result-to-context formatting.

Provides :class:`WebSearcher` that wraps the ``duckduckgo_search`` library
and formats results as RAG context blocks with source URLs, suitable for
injecting into the generator prompt along with KB contexts.

Usage::

    from src.search.web_searcher import WebSearcher

    searcher = WebSearcher()
    response = await searcher.search("fiber laser latest advances 2025")
    print(response.to_context_block())
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ddgs import DDGS

if TYPE_CHECKING:
    pass

logger = logging.getLogger("laser-rag.web_searcher")


# ===========================================================================
#  Dataclasses
# ===========================================================================


@dataclass
class WebSearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str
    index: int = 0


@dataclass
class WebSearchResponse:
    """Collection of web search results with formatting helpers."""

    query: str
    results: list[WebSearchResult] = field(default_factory=list)
    error: str | None = None

    # ------------------------------------------------------------------
    #  context formatting
    # ------------------------------------------------------------------

    def to_context_block(self, max_results: int = 5) -> str:
        """Format web results as a prompt-ready context string.

        Each result is labeled with a source tag and URL so the LLM can
        cite it in the final answer.

        Parameters
        ----------
        max_results:
            Maximum number of results to include.

        Returns
        -------
        str
            Formatted context block, or a placeholder string when no
            results are available.
        """
        if not self.results:
            return "（网络搜索未找到相关结果）"

        parts: list[str] = [
            "以下是从网络搜索获得的参考资料，请优先使用这些信息：",
            "",
        ]
        for r in self.results[:max_results]:
            parts.append(
                f"--- 🌐 网络来源 {r.index}：「{r.title}」 ---\n"
                f"URL: {r.url}\n"
                f"{r.snippet}"
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    #  reference conversion — for Generator citation extraction
    # ------------------------------------------------------------------

    def to_references(self) -> list[dict]:
        """Convert results to a list of dicts matching the Reference schema.

        Returns
        -------
        list[dict]
            Each dict has keys ``index``, ``source``, ``text``, ``url``,
            and ``is_web`` suitable for the SSE ``references`` event.
        """
        return [
            {
                "index": r.index,
                "source": r.title,
                "text": r.snippet[:200],
                "url": r.url,
                "is_web": True,
            }
            for r in self.results
        ]


# ===========================================================================
#  WebSearcher
# ===========================================================================


class WebSearcher:
    """DuckDuckGo-based web search for filling knowledge gaps.

    The synchronous ``DDGS().text()`` call is offloaded to a thread-pool
    executor so that it never blocks the async event loop.

    Parameters
    ----------
    max_results:
        Default number of search results to return (1–10).
    region:
        Region code for DuckDuckGo (e.g. ``"wt-wt"`` for worldwide,
        ``"cn-zh"`` for China Chinese).
    timelimit:
        Optional time filter — ``"d"`` (day), ``"w"`` (week), ``"m"``
        (month), ``"y"`` (year).
    """

    def __init__(
        self,
        max_results: int = 5,
        region: str = "wt-wt",
        timelimit: str | None = None,
        timeout: int = 15,
    ) -> None:
        self.max_results = max(max_results, 1)
        self.region = region
        self.timelimit = timelimit
        self.timeout = timeout

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------

    async def search(
        self, query: str, max_results: int | None = None
    ) -> WebSearchResponse:
        """Execute a DuckDuckGo text search asynchronously.

        Parameters
        ----------
        query:
            The search query string.
        max_results:
            Override the default result count.

        Returns
        -------
        WebSearchResponse
            Wraps the search results (or error information).
        """
        n = max_results or self.max_results
        loop = asyncio.get_running_loop()

        def _sync_search() -> list[dict]:
            with DDGS(timeout=self.timeout) as ddgs:
                return list(
                    ddgs.text(
                        query,
                        region=self.region,
                        safesearch="moderate",
                        timelimit=self.timelimit,
                        max_results=n,
                    )
                )

        try:
            raw_results = await loop.run_in_executor(None, _sync_search)
        except Exception as exc:
            logger.exception("Web search failed for %r", query[:60])
            return WebSearchResponse(query=query, error=str(exc))

        results: list[WebSearchResult] = []
        for i, r in enumerate(raw_results):
            title = r.get("title", "")
            url = r.get("href", "")
            body = r.get("body", "")
            # skip results that are completely empty
            if not body and not title:
                continue
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=body,
                    index=i + 1,
                )
            )

        logger.info(
            "Web search %r → %d result(s)", query[:60], len(results)
        )
        return WebSearchResponse(query=query, results=results)

    # ------------------------------------------------------------------
    #  sync convenience
    # ------------------------------------------------------------------

    def search_sync(
        self, query: str, max_results: int | None = None
    ) -> WebSearchResponse:
        """Synchronous version of :meth:`search` — for CLI / debugging."""
        n = max_results or self.max_results

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw_results = list(
                    ddgs.text(
                        query,
                        region=self.region,
                        safesearch="moderate",
                        timelimit=self.timelimit,
                        max_results=n,
                    )
                )
        except Exception as exc:
            logger.exception("Web search failed for %r", query[:60])
            return WebSearchResponse(query=query, error=str(exc))

        results: list[WebSearchResult] = []
        for i, r in enumerate(raw_results):
            title = r.get("title", "")
            url = r.get("href", "")
            body = r.get("body", "")
            if not body and not title:
                continue
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=body,
                    index=i + 1,
                )
            )

        logger.info("Web search %r → %d result(s)", query[:60], len(results))
        return WebSearchResponse(query=query, results=results)
