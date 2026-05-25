"""LLM call wrapper that records model / tokens / cost / latency / cache-hit (SPEC §6.9).

Wrap any LLM invocation so every call produces an :class:`LLMCallRecord` and an
OTel span. Cost is derived from a per-model price table (USD per 1M tokens);
cache reads are billed at the discounted Anthropic rate when reported.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from opentelemetry import trace

from common.types import LLMCallRecord
from harness.observability.logging import get_logger

_tracer = trace.get_tracer("knowledge-agent.llm")
_log = get_logger("harness.observability.llm")


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens for input / output / cached-input reads."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float


# Approximate public list prices (USD / 1M tokens). Update as pricing changes;
# keep this the single source of truth for cost accounting.
PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0, 0.30),
    "claude-haiku-4-5-20251001": ModelPricing(1.0, 5.0, 0.10),
}

_FALLBACK_PRICING = ModelPricing(3.0, 15.0, 0.30)


def compute_cost(
    model: str, tokens_in: int, tokens_out: int, *, cached_tokens_in: int = 0
) -> float:
    """Compute USD cost for a call, billing cached input at the discounted rate."""
    pricing = PRICING.get(model, _FALLBACK_PRICING)
    fresh_in = max(0, tokens_in - cached_tokens_in)
    return (
        fresh_in * pricing.input_per_mtok
        + cached_tokens_in * pricing.cached_input_per_mtok
        + tokens_out * pricing.output_per_mtok
    ) / 1_000_000.0


@dataclass
class LLMResponse:
    """Normalized result returned by an instrumented LLM call."""

    text: str
    record: LLMCallRecord
    raw: Any = None


def _extract_usage(raw: Any) -> tuple[int, int, int]:
    """Best-effort extraction of (tokens_in, tokens_out, cached_tokens_in).

    Handles Anthropic-style ``usage`` objects/dicts; returns zeros if absent so
    the wrapper degrades gracefully for stubbed or non-Anthropic clients.
    """
    usage = getattr(raw, "usage", None)
    if usage is None and isinstance(raw, dict):
        usage = raw.get("usage")
    if usage is None:
        return 0, 0, 0

    def _field(name: str) -> int:
        if isinstance(usage, dict):
            return int(usage.get(name, 0) or 0)
        return int(getattr(usage, name, 0) or 0)

    tokens_in = _field("input_tokens")
    tokens_out = _field("output_tokens")
    cached = _field("cache_read_input_tokens")
    return tokens_in, tokens_out, cached


async def instrumented_call(
    model: str,
    fn: Callable[[], Awaitable[Any]],
    *,
    trace_id: UUID | None = None,
    extract_text: Callable[[Any], str] | None = None,
) -> LLMResponse:
    """Invoke ``fn`` (an async LLM call), recording full telemetry.

    ``fn`` should return the provider's raw response; ``extract_text`` pulls the
    text out of it (defaults to ``str``). Usage/cost are read from the response's
    ``usage`` block when present.
    """
    start = time.perf_counter()
    with _tracer.start_as_current_span(f"llm.{model}") as span:
        span.set_attribute("llm.model", model)
        raw = await fn()
        latency_ms = (time.perf_counter() - start) * 1000.0

        tokens_in, tokens_out, cached_in = _extract_usage(raw)
        cost = compute_cost(model, tokens_in, tokens_out, cached_tokens_in=cached_in)
        cache_hit = cached_in > 0

        record = LLMCallRecord(
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            trace_id=trace_id,
        )

        span.set_attribute("llm.tokens_in", tokens_in)
        span.set_attribute("llm.tokens_out", tokens_out)
        span.set_attribute("llm.cost_usd", cost)
        span.set_attribute("llm.latency_ms", latency_ms)
        span.set_attribute("llm.cache_hit", cache_hit)
        _log.info(
            "llm.call",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 2),
            cache_hit=cache_hit,
        )

        text = (extract_text or str)(raw)
        return LLMResponse(text=text, record=record, raw=raw)


__all__ = [
    "ModelPricing",
    "PRICING",
    "compute_cost",
    "LLMResponse",
    "instrumented_call",
]
