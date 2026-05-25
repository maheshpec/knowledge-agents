"""The ``@traced`` decorator (SPEC §6.9).

Wraps any sync or async function in an OpenTelemetry span and emits a structured
log event on entry/exit (incl. latency and exceptions). LangSmith tracing is
layered on via its own ``@traceable`` decorator at call sites; this decorator is
the always-on, dependency-light baseline so every state transition is captured
even when LangSmith is not configured.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from opentelemetry import trace

from harness.observability.logging import get_logger

P = ParamSpec("P")
R = TypeVar("R")

_tracer = trace.get_tracer("knowledge-agent")
_log = get_logger("harness.observability")


def traced(
    span_name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator factory wrapping a callable in an OTel span + structured log.

    Works on both coroutine and plain functions. Records latency and re-raises
    any exception after marking the span as errored.

    Example::

        @traced(span_name="retrieval.hybrid")
        async def hybrid_retrieve(query: Query) -> RetrievalResult: ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        name = span_name or f"{func.__module__}.{func.__qualname__}"
        base_attrs = attributes or {}

        if _is_coroutine(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start = time.perf_counter()
                with _tracer.start_as_current_span(name) as span:
                    _set_attrs(span, base_attrs)
                    try:
                        result = await func(*args, **kwargs)  # type: ignore[misc]
                    except Exception as exc:
                        _record_error(span, name, exc, start)
                        raise
                    _record_ok(span, name, start)
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            with _tracer.start_as_current_span(name) as span:
                _set_attrs(span, base_attrs)
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    _record_error(span, name, exc, start)
                    raise
                _record_ok(span, name, start)
                return result

        return sync_wrapper

    return decorator


def _is_coroutine(func: Callable[..., Any]) -> bool:
    import asyncio

    return asyncio.iscoroutinefunction(func)


def _set_attrs(span: trace.Span, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        span.set_attribute(key, value)


def _record_ok(span: trace.Span, name: str, start: float) -> None:
    latency_ms = (time.perf_counter() - start) * 1000.0
    span.set_attribute("latency_ms", latency_ms)
    _log.debug("span.ok", span=name, latency_ms=round(latency_ms, 2))


def _record_error(span: trace.Span, name: str, exc: Exception, start: float) -> None:
    latency_ms = (time.perf_counter() - start) * 1000.0
    span.set_attribute("latency_ms", latency_ms)
    span.record_exception(exc)
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
    _log.error("span.error", span=name, error=str(exc), error_type=type(exc).__name__)


__all__ = ["traced"]
