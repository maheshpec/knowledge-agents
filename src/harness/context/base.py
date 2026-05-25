"""Context-packing contracts (SPEC §6.6).

The packer decides *what goes into the next LLM call and in what order*: system
prompt, skills, memory hits, retrieved chunks, scratchpad, recent turns — subject
to a token budget. This is distinct from compaction (which shrinks history).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.messages import BaseMessage

from common.schemas import RetrievalResult

# The canonical Skill model lives in common.types (SPEC §6.8); re-exported here so
# the packer signature and existing imports (`from harness.context import Skill`)
# stay stable now that the Phase 2 loader produces these.
from common.types import MemoryItem, Skill


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for budget enforcement.

    Deliberately dependency-free; exact accounting happens server-side. Slightly
    conservative so the packer trims before the real limit, not after.
    """
    return max(1, len(text) // 4)


@runtime_checkable
class ContextPacker(Protocol):
    """Assemble the message list for the next LLM call (SPEC §6.6)."""

    def pack(
        self,
        system: str,
        skills: list[Skill],
        memory_hits: list[MemoryItem],
        retrieval: RetrievalResult | None,
        scratchpad: str,
        messages: list[BaseMessage],
        budget_tokens: int,
    ) -> list[BaseMessage]: ...


__all__ = ["Skill", "ContextPacker", "estimate_tokens"]
