"""Compaction contracts (SPEC §6.5).

Compaction shrinks an :class:`OrchestratorState` when its history approaches the
model limit, while preserving what the agent needs to continue coherently:
the original goal, current plan, the last few turns, and all citations. It drops
raw tool outputs, uncited retrieval candidates, and exploratory branches.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from harness.context.base import estimate_tokens
from harness.orchestrator.state import OrchestratorState

# Default trigger: ~120k tokens (mirrors configs/default.yaml compaction_threshold).
DEFAULT_COMPACTION_THRESHOLD = 120_000


class CompactionConfig(BaseModel):
    """Tunable compaction policy (SPEC §6.5)."""

    max_tokens: int = DEFAULT_COMPACTION_THRESHOLD
    keep_last_turns: int = 3  # a turn ≈ a (human, assistant) message pair


@runtime_checkable
class Compactor(Protocol):
    """Decide whether to compact, and do it (SPEC §6.5)."""

    name: str

    async def should_compact(self, state: OrchestratorState) -> bool: ...
    async def compact(self, state: OrchestratorState) -> OrchestratorState: ...


def message_text(message: BaseMessage) -> str:
    """Extract plain text from a message whose content may be blocks or a string."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return " ".join(p for p in parts if p)
    return str(content)


def estimate_state_tokens(state: OrchestratorState) -> int:
    """Rough token estimate of the conversation history in a state."""
    messages = state.get("messages", []) or []
    return sum(estimate_tokens(message_text(m)) for m in messages)


def split_keep_tail(messages: list[BaseMessage], keep_last_turns: int) -> tuple[list, list]:
    """Split messages into (dropped_prefix, kept_tail), keeping the last turns.

    A turn is approximated as two messages (human + assistant), so we keep the
    last ``keep_last_turns * 2`` messages.
    """
    keep_n = max(0, keep_last_turns) * 2
    if keep_n == 0 or keep_n >= len(messages):
        return ([], list(messages)) if keep_n >= len(messages) else (list(messages), [])
    return list(messages[:-keep_n]), list(messages[-keep_n:])


__all__ = [
    "DEFAULT_COMPACTION_THRESHOLD",
    "CompactionConfig",
    "Compactor",
    "message_text",
    "estimate_state_tokens",
    "split_keep_tail",
]
