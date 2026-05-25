"""Shared lightweight types used across harness and index modules (SPEC §6).

Distinct from `schemas.py`: these are the supporting value types (budgets, tool
calls, memory items, enums) rather than the core data-flow models.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MimeType(StrEnum):
    """Supported ingestion source types (SPEC §7.1)."""

    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    HTML = "text/html"
    MARKDOWN = "text/markdown"
    PLAIN = "text/plain"
    CODE = "text/x-code"
    UNKNOWN = "application/octet-stream"


# --- Budget (SPEC §6.11) ---------------------------------------------------


class BudgetSpec(BaseModel):
    """A budget allocation for a request or sub-agent."""

    max_usd: float = 1.0
    max_tokens: int | None = None
    max_tool_calls: int | None = None


class BudgetGrant(BaseModel):
    """A reservation handed out by the BudgetTracker; settled via `consume`."""

    grant_id: UUID = Field(default_factory=uuid4)
    amount: float  # reserved USD
    settled: bool = False


# --- Tools (SPEC §6.1, §6.7) ----------------------------------------------


class ToolCall(BaseModel):
    """A pending or executed tool invocation tracked in orchestrator state."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result: Any = None


class ToolResult(BaseModel):
    """The outcome of a sandboxed tool execution."""

    tool: str
    ok: bool
    output: Any = None
    error: str | None = None
    result_hash: str | None = None  # for observability (SPEC §6.9)


# --- Memory (SPEC §6.3) ----------------------------------------------------


class MemoryItem(BaseModel):
    """A unit stored in / retrieved from any memory scope."""

    key: str
    value: Any
    scope: Literal["working", "session", "long_term"] = "working"
    score: float | None = None  # similarity score when read from a vector scope
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- LLM telemetry (SPEC §6.9) --------------------------------------------


class LLMCallRecord(BaseModel):
    """Recorded for every LLM call: model, tokens, cost, latency, cache hit."""

    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cache_hit: bool = False
    trace_id: UUID | None = None


__all__ = [
    "MimeType",
    "BudgetSpec",
    "BudgetGrant",
    "ToolCall",
    "ToolResult",
    "MemoryItem",
    "LLMCallRecord",
]
