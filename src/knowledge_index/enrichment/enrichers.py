"""Concrete enrichers (SPEC §7.3).

``ContextualEnricher`` implements Anthropic-style contextual retrieval: the full
document is sent as a *cached* prompt prefix (one cache write, reused for every
chunk of that document) and only the per-chunk tail varies — this is what keeps
enrichment under the SPEC §7.3 cost target. ``Title``/``Summary``/``Null`` are
cheaper alternatives and baselines.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from common.schemas import Chunk
from harness.cache.prompt_cache import cacheable_text_block
from knowledge_index.ingestion.base import ParsedDoc

# The exact prompt template from SPEC §7.3 (do not reword — it is benchmarked).
_CONTEXT_INSTRUCTION = (
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short (max 100 tokens) succinct context to situate this chunk within\n"
    "the overall document, for the purposes of improving search retrieval of the\n"
    "chunk. Answer only with the succinct context and nothing else."
)

# A completion fn takes a list of content blocks and returns the model's text.
CompletionFn = Callable[[list[dict[str, Any]]], Awaitable[str]]


class NullEnricher:
    """Baseline: no enrichment (SPEC §7.3)."""

    name = "null"

    async def enrich(self, doc: ParsedDoc, chunks: list[Chunk]) -> list[Chunk]:
        return chunks


class TitleEnricher:
    """Prepend doc title + section header path (no LLM, no network)."""

    name = "title"

    async def enrich(self, doc: ParsedDoc, chunks: list[Chunk]) -> list[Chunk]:
        title = doc.metadata.get("title")
        for c in chunks:
            parts: list[str] = []
            if title:
                parts.append(str(title))
            header_path = c.metadata.get("header_path") or []
            parts.extend(str(h) for h in header_path if h)
            if parts:
                c.context = " › ".join(parts)
        return chunks


class ContextualEnricher:
    """LLM-generated per-chunk context with the full doc cached (SPEC §7.3)."""

    name = "contextual"

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5-20251001",
        max_context_tokens: int = 80,
        max_concurrency: int = 8,
        api_key: str | None = None,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_concurrency = max_concurrency
        self._api_key = api_key
        self._completion_fn = completion_fn
        self._client = None
        self.config: dict[str, Any] = {"max_context_tokens": max_context_tokens}

    def _default_completion_fn(self) -> CompletionFn:
        async def _complete(blocks: list[dict[str, Any]]) -> str:
            if self._client is None:
                from anthropic import AsyncAnthropic  # type: ignore

                self._client = AsyncAnthropic(api_key=self._api_key)
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_context_tokens + 20,
                messages=[{"role": "user", "content": blocks}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

        return _complete

    def _blocks(self, doc_text: str, chunk_text: str) -> list[dict[str, Any]]:
        # Document block is cached (constant across all chunks of this doc);
        # the per-chunk instruction tail is not cached.
        return [
            cacheable_text_block(f"<document>\n{doc_text}\n</document>", cache=True),
            cacheable_text_block(
                _CONTEXT_INSTRUCTION.format(chunk=chunk_text), cache=False
            ),
        ]

    async def enrich(self, doc: ParsedDoc, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks
        complete = self._completion_fn or self._default_completion_fn()
        sem = asyncio.Semaphore(self.max_concurrency)

        async def one(chunk: Chunk) -> None:
            async with sem:
                ctx = await complete(self._blocks(doc.text, chunk.text))
            chunk.context = ctx.strip() or None

        await asyncio.gather(*(one(c) for c in chunks))
        return chunks


class SummaryEnricher:
    """Prepend a one-sentence LLM summary of each chunk (SPEC §7.3)."""

    name = "summary"

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5-20251001",
        max_concurrency: int = 8,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        self.model = model
        self.max_concurrency = max_concurrency
        self._completion_fn = completion_fn

    async def enrich(self, doc: ParsedDoc, chunks: list[Chunk]) -> list[Chunk]:
        if self._completion_fn is None:
            from anthropic import AsyncAnthropic  # type: ignore

            client = AsyncAnthropic()

            async def _complete(blocks: list[dict[str, Any]]) -> str:
                resp = await client.messages.create(
                    model=self.model, max_tokens=60,
                    messages=[{"role": "user", "content": blocks}],
                )
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                ).strip()

            complete = _complete
        else:
            complete = self._completion_fn
        sem = asyncio.Semaphore(self.max_concurrency)

        async def one(chunk: Chunk) -> None:
            async with sem:
                blocks = [
                    cacheable_text_block(
                        "Summarize the following text in one short sentence.\n\n"
                        f"{chunk.text}",
                        cache=False,
                    )
                ]
                chunk.context = (await complete(blocks)).strip() or None

        await asyncio.gather(*(one(c) for c in chunks))
        return chunks


__all__ = ["NullEnricher", "TitleEnricher", "ContextualEnricher", "SummaryEnricher"]
