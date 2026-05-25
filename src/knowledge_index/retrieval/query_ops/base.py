"""QueryOp protocol, the sequential apply helper, and a default LLM completer (SPEC §7.6.2).

Query ops transform a :class:`Query` before retrieval (rewrite, HyDE, decompose,
step-back). They compose as an ordered list — each receives the output of the
previous — so ``[Rewriter(), HyDEExpander()]`` rewrites then expands.

LLM-backed ops take an injected ``complete`` callable (``str -> str``) so they run
offline under test. :func:`default_completer` builds a real one from
``langchain_anthropic.ChatAnthropic`` lazily, on first use.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from common.schemas import Query
from harness.observability.tracing import traced

CompleteFn = Callable[[str], Awaitable[str]]

# Cheap, fast model is the right default for query reformulation.
DEFAULT_QUERY_OP_MODEL = "claude-haiku-4-5-20251001"


@runtime_checkable
class QueryOp(Protocol):
    """Transform a query prior to retrieval (SPEC §7.6.2)."""

    name: str

    async def transform(self, query: Query) -> Query: ...


def default_completer(model: str = DEFAULT_QUERY_OP_MODEL) -> CompleteFn:
    """Build a ``str -> str`` async completer backed by ChatAnthropic (lazy import)."""

    async def _complete(prompt: str) -> str:
        from langchain_anthropic import ChatAnthropic

        from common.settings import get_settings

        # Build kwargs as a dict: ChatAnthropic coerces api_key str -> SecretStr and
        # accepts model/max_tokens at runtime, but its type stubs disagree.
        init_kwargs: dict[str, Any] = {
            "model": model,
            "api_key": get_settings().anthropic_api_key,
            "max_tokens": 512,
        }
        llm = ChatAnthropic(**init_kwargs)
        response = await llm.ainvoke(prompt)
        content = response.content
        return content if isinstance(content, str) else str(content)

    return _complete


@traced(span_name="retrieval.query_ops.apply")
async def apply_query_ops(ops: list[QueryOp], query: Query) -> Query:
    """Apply each op in order, threading the transformed query through the chain."""
    for op in ops:
        query = await op.transform(query)
    return query


__all__ = [
    "CompleteFn",
    "DEFAULT_QUERY_OP_MODEL",
    "QueryOp",
    "default_completer",
    "apply_query_ops",
]
