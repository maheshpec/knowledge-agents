"""Memory layers (SPEC §6.3): working, session, long-term + extraction."""

from __future__ import annotations

from harness.memory.base import Memory, MemoryScope, ScopeStore
from harness.memory.extraction import (
    ExtractedFact,
    ExtractFn,
    ExtractionResult,
    MemoryExtractor,
)
from harness.memory.longterm import DEFAULT_MEMORY_COLLECTION, LongTermMemory
from harness.memory.manager import LayeredMemory
from harness.memory.session import SessionMemory
from harness.memory.working import WorkingMemory

__all__ = [
    "Memory",
    "MemoryScope",
    "ScopeStore",
    "WorkingMemory",
    "SessionMemory",
    "LongTermMemory",
    "DEFAULT_MEMORY_COLLECTION",
    "LayeredMemory",
    "MemoryExtractor",
    "ExtractedFact",
    "ExtractionResult",
    "ExtractFn",
]
