"""LLM generator with streaming SSE output and citation extraction.

Provides :class:`Generator` which calls the OpenAI Chat Completion API
with streaming enabled, yields SSE events chunk by chunk, extracts
citation markers from the generated answer, and attaches a reference
list in the final response.

Usage::

    from src.generation.generator import Generator
    from src.generation.prompt import build_prompt

    generator = Generator()
    messages = build_prompt(question, contexts)

    # server-side (FastAPI SSE endpoint)
    async def sse_endpoint():
        async for event in generator.generate(messages, contexts):
            yield event

    # or sync (for debugging / CLI)
    for event in generator.generate_sync(messages, contexts):
        print(event)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator, Generator, Sequence, Union

from openai import AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from config import Settings
    from src.retrieval.reranker import RerankerResult

logger = logging.getLogger("laser-rag.generator")

# ---------------------------------------------------------------------------
#  output dataclass
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """A citation reference extracted from the generated answer.

    Attributes
    ----------
    index:  Citation number (e.g. ``1`` in ``[1]``), 1-based.
    source: Filename or source identifier.
    page:   Page number if available.
    text:   Snippet of the referenced context text (first 200 chars).
    url:    URL for web-sourced references (🌐 network supplement).
    is_web: Whether this reference is from web supplement (🌐) vs local KB.
    """

    index: int
    source: str
    text: str
    page: int | None = None
    url: str | None = None
    is_web: bool = False


# ---------------------------------------------------------------------------
#  SSE event helpers
# ---------------------------------------------------------------------------

_SSE_DELIMITER = "\n\n"


def _sse_event(event: str, data: dict | str) -> str:
    """Format a single SSE event string.

    >>> _sse_event("token", {"content": "你好"})
    'event: token\\ndata: {"content":"你好"}\\n\\n'
    """
    if isinstance(data, dict):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = data
    return f"event: {event}\ndata: {payload}{_SSE_DELIMITER}"


# ===========================================================================
#  Generator
# ===========================================================================


class Generator:
    """LLM generator with streaming SSE output and citation extraction.

    Calls the OpenAI-compatible Chat Completion API, streams the response
    token-by-token as SSE ``token`` events, then extracts citation markers
    and emits a ``references`` event with a structured bibliography.

    Parameters
    ----------
    model:
        Model name.  Default from ``GENERATOR_MODEL_NAME`` config,
        falling back to ``"gpt-4o"``.
    api_key:
        API key.  Default from ``LLM_API_KEY`` config.
    api_base:
        API base URL.  Default from ``LLM_API_BASE`` config.
    temperature:
        LLM temperature.  Default from ``LLM_TEMPERATURE`` config.
    max_tokens:
        Max tokens to generate.  Default from ``LLM_MAX_TOKENS`` config.
    timeout:
        Per-request timeout in seconds.  Default from ``LLM_TIMEOUT`` config.

    SSE Event Reference
    -------------------
    =============== ==================================================
    Event            ``data`` payload
    =============== ==================================================
    ``token``        ``{"content": "文"}`` — incremental text chunk
    ``references``   ``{"references": [{"index": 1, "source": "...",
                     "text": "...", "page": 3}, ...]}``
    ``done``         ``{"finish_reason": "stop", "total_tokens": 512}``
    ``error``        ``{"message": "..."}`` — may include
                     ``partial_content`` if some output was already sent
    =============== ==================================================
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        from config import settings

        self._s: Settings = settings

        # model: explicit arg → GENERATOR_MODEL_NAME config → hard-coded fallback
        self.model = (
            model
            or self._s.generator_model_name
            or "deepseek-v4-pro"
        )
        self.temperature = (
            temperature if temperature is not None else self._s.llm_temperature
        )
        self.max_tokens = (
            max_tokens if max_tokens is not None else self._s.llm_max_tokens
        )
        self.timeout = (
            timeout if timeout is not None else self._s.llm_timeout
        )

        _key = api_key or self._s.llm_api_key
        _base = api_base or self._s.llm_api_base

        if not _key:
            raise ValueError(
                "LLM API key not set. Provide it via LLM_API_KEY "
                "or OPENAI_API_KEY environment variable."
            )

        self._client = OpenAI(api_key=_key, base_url=_base, timeout=self.timeout)
        self._async_client = AsyncOpenAI(
            api_key=_key, base_url=_base, timeout=self.timeout
        )

    # ------------------------------------------------------------------
    #  public API — async streaming
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, str]],
        contexts: Sequence[Union[str, RerankerResult]] | None = None,
        *,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate answer via LLM, yielding SSE events asynchronously.

        Parameters
        ----------
        messages:
            Structured message list from :func:`~.prompt.build_prompt`.
        contexts:
            Original context chunks that were injected into *messages*.
            Used for citation extraction — ``[1]`` maps to
            ``contexts[0]``, ``[2]`` to ``contexts[1]``, etc.
            When ``None``, no ``references`` event is emitted.
        stream:
            If ``True`` (default), yield incremental ``token`` events.
            If ``False``, make a single non-streaming API call and
            yield one ``token`` event with the full content.

        Yields
        ------
        str
            SSE-formatted event strings.  See the class docstring for
            the event reference.
        """
        if not stream:
            # non-streaming async — handled inline below
            try:
                response = await self._async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
            except Exception as exc:
                logger.exception("Async non-streaming generation failed")
                yield _sse_event("error", {"message": str(exc)})
                return

            content: str = response.choices[0].message.content or ""
            yield _sse_event("token", {"content": content})

            if contexts:
                refs = self.extract_citations(content, contexts)
                if refs:
                    yield _sse_event("references", {
                        "references": [self._ref_to_dict(r) for r in refs],
                    })

            finish = response.choices[0].finish_reason or "stop"
            yield _sse_event("done", {"finish_reason": finish})
            return

        full_text = ""
        finish_reason = "stop"

        try:
            stream_response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            async for chunk in stream_response:
                choices = chunk.choices
                if not choices:
                    continue

                delta = choices[0].delta
                finish = choices[0].finish_reason

                if delta and delta.content:
                    full_text += delta.content
                    yield _sse_event("token", {"content": delta.content})

                if finish:
                    finish_reason = finish

            # ---- post-streaming: citation extraction ----
            if contexts:
                refs = self.extract_citations(full_text, contexts)
                if refs:
                    yield _sse_event("references", {
                        "references": [self._ref_to_dict(r) for r in refs],
                    })

            yield _sse_event("done", {
                "finish_reason": finish_reason,
                "total_tokens": len(full_text),
            })

        except Exception as exc:
            logger.exception("Streaming generation failed")
            if full_text:
                yield _sse_event("error", {
                    "message": str(exc),
                    "partial_content": full_text,
                })
            else:
                yield _sse_event("error", {"message": str(exc)})

    # ------------------------------------------------------------------
    #  public API — sync streaming
    # ------------------------------------------------------------------

    def generate_sync(
        self,
        messages: list[dict[str, str]],
        contexts: Sequence[Union[str, RerankerResult]] | None = None,
        *,
        stream: bool = True,
    ) -> Generator[str, None, None]: # type: ignore
        """Synchronous version of :meth:`generate`.

        Yields the same SSE event strings.  Use this in CLI tools,
        debugging, or when the event loop is not available.
        """
        if not stream:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
            except Exception as exc:
                logger.exception("Sync non-streaming generation failed")
                yield _sse_event("error", {"message": str(exc)})
                return

            content: str = response.choices[0].message.content or ""
            yield _sse_event("token", {"content": content})

            if contexts:
                refs = self.extract_citations(content, contexts)
                if refs:
                    yield _sse_event("references", {
                        "references": [self._ref_to_dict(r) for r in refs],
                    })

            finish = response.choices[0].finish_reason or "stop"
            yield _sse_event("done", {"finish_reason": finish})
            return

        full_text = ""
        finish_reason = "stop"

        try:
            stream_response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in stream_response:
                choices = chunk.choices
                if not choices:
                    continue

                delta = choices[0].delta
                finish = choices[0].finish_reason

                if delta and delta.content:
                    full_text += delta.content
                    yield _sse_event("token", {"content": delta.content})

                if finish:
                    finish_reason = finish

            if contexts:
                refs = self.extract_citations(full_text, contexts)
                if refs:
                    yield _sse_event("references", {
                        "references": [self._ref_to_dict(r) for r in refs],
                    })

            yield _sse_event("done", {
                "finish_reason": finish_reason,
                "total_tokens": len(full_text),
            })

        except Exception as exc:
            logger.exception("Streaming generation failed")
            if full_text:
                yield _sse_event("error", {
                    "message": str(exc),
                    "partial_content": full_text,
                })
            else:
                yield _sse_event("error", {"message": str(exc)})

    # ------------------------------------------------------------------
    #  citation extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_citations(
        text: str,
        contexts: Sequence[Union[str, RerankerResult]],
    ) -> list[Reference]:
        """Extract citation markers from *text* and map them to *contexts*.

        Supports two citation styles simultaneously:

        **Bracket style** (numeric reference):
          ``[1]``, ``[2]``, ``[1,2,3]`` — the number is a 1-based index
          into *contexts*.

        **Full style** (inline source):
          ``【来源：Nd:YAG激光器技术手册】`` or
          ``【来源：Nd:YAG激光器技术手册，第3页】`` — matched against
          the filenames in *contexts*.

        Parameters
        ----------
        text:
            The generated answer text to scan.
        contexts:
            The context chunks that were injected into the prompt.
            Order matters — ``[1]`` maps to ``contexts[0]``, etc.

        Returns
        -------
        list[Reference]
            Unique references sorted by index.  Empty list when
            *contexts* is empty or no citations are found.
        """
        if not contexts:
            return []

        # ---- normalise contexts to uniform dicts ----
        normalized: list[dict] = []
        for ctx in contexts:
            if isinstance(ctx, str):
                normalized.append({
                    "source": "未知来源",
                    "page": None,
                    "text": ctx[:200],
                })
            else:
                normalized.append({
                    "source": ctx.filename or "未知来源",
                    "page": ctx.page,
                    "text": ctx.text[:200],
                })

        refs: dict[int, Reference] = {}

        # ---- style 1: bracket [1], [2], [1,2,3] ----
        for match in re.finditer(r"\[([\d,\s]+)\]", text):
            nums = re.findall(r"\d+", match.group(1))
            for num_str in nums:
                n = int(num_str)
                if 1 <= n <= len(normalized) and n not in refs:
                    ctx = normalized[n - 1]
                    refs[n] = Reference(
                        index=n,
                        source=ctx["source"],
                        page=ctx.get("page"),  # type: ignore[arg-type]
                        text=ctx["text"],
                    )

        # ---- style 2: full 【来源：filename】 or 【来源：filename，第X页】 ----
        # Also matches 【filename.docx】 style when the LLM omits the "来源：" prefix
        for match in re.finditer(r"【(?:来源[：:]\s*)?([^，】\d]+)(?:，第(\d+)页)?】", text):
            source = match.group(1).strip()
            page = int(match.group(2)) if match.group(2) else None
            # link to context by filename
            for i, ctx in enumerate(normalized):
                if ctx["source"] == source:
                    idx = i + 1
                    if idx not in refs:
                        refs[idx] = Reference(
                            index=idx,
                            source=source,
                            page=page or ctx.get("page"),  # type: ignore[arg-type]
                            text=ctx["text"],
                        )
                    break

        # ---- style 3: end-of-answer reference list ----
        # Matches the format:
        #   > 📚 参考文档：
        #   > 1. 《光纤激光器原理.docx》 — 第3页
        #   > 2. 《Nd:YAG技术手册》
        for match in re.finditer(
            r"(?:> )?\d+\.\s*《([^》]+)》\s*(?:[—\-–]\s*(?:第(\d+)(?:\s*[–\-]\s*(\d+))?页)?)?",
            text,
        ):
            source = match.group(1).strip()
            page = None
            if match.group(2):
                page = int(match.group(2))
            # match against known contexts by filename
            for i, ctx in enumerate(normalized):
                if ctx["source"] == source or source in ctx["source"] or ctx["source"] in source:
                    idx = i + 1
                    if idx not in refs:
                        refs[idx] = Reference(
                            index=idx,
                            source=ctx["source"],
                            page=page or ctx.get("page"),  # type: ignore[arg-type]
                            text=ctx["text"],
                        )
                    break

        # ---- style 4: web references with 🌐 marker and URL ----
        # Matches:
        #   > 2. 🌐 《Wikipedia - Fiber laser》 — https://en.wikipedia.org/wiki/Fiber_laser
        #   > 3. 🌐 《TRUMPF Laser Cutting》 — https://www.trumpf.com/... — 第5页
        # Also matches without the leading "> " prefix
        for match in re.finditer(
            r"(?:> )?(\d+)\.\s*🌐\s*《([^》]+)》\s*[—\-–]\s*(https?://[^\s)]+)"
            r"(?:\s*[—\-–]\s*(?:第(\d+)(?:\s*[–\-]\s*(\d+))?页))?",
            text,
        ):
            idx = int(match.group(1))
            source = match.group(2).strip()
            url = match.group(3).strip()
            page = None
            if match.group(4):
                page = int(match.group(4))

            refs[idx] = Reference(
                index=idx,
                source=source,
                url=url,
                is_web=True,
                page=page,
                text=f"🌐 网络补充来源：{source} ({url})",
            )

        return sorted(refs.values(), key=lambda r: r.index)

    # ------------------------------------------------------------------
    #  internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ref_to_dict(ref: Reference) -> dict:
        """Convert :class:`Reference` to a JSON-serializable dict."""
        d: dict = {
            "index": ref.index,
            "source": ref.source,
            "text": ref.text,
        }
        if ref.page is not None:
            d["page"] = ref.page
        if ref.url is not None:
            d["url"] = ref.url
        if ref.is_web:
            d["is_web"] = ref.is_web
        return d

    def generate_raw(
        self,
        messages: list[dict[str, str]],
        contexts: Sequence[Union[str, RerankerResult]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Convenience wrapper for one-shot non-streaming generation.

        Returns the raw text content — no SSE wrapping.  Useful for
        programmatic consumption where SSE parsing is not needed.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.exception("Generation failed")
            raise
