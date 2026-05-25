"""structlog JSON logging configuration (SPEC §6.9).

Call ``configure_logging()`` once at process start. Every module then uses
``get_logger(__name__)`` to emit structured JSON events that interleave cleanly
with OpenTelemetry spans and LangSmith traces.
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog for JSON (prod) or console (dev) output. Idempotent."""
    global _configured
    if _configured:
        return

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level))

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


__all__ = ["configure_logging", "get_logger"]
