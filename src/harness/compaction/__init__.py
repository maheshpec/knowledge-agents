"""Compaction (SPEC §6.5): shrink orchestrator state while preserving coherence."""

from __future__ import annotations

from harness.compaction.base import (
    DEFAULT_COMPACTION_THRESHOLD,
    CompactionConfig,
    Compactor,
    estimate_state_tokens,
    message_text,
    split_keep_tail,
)
from harness.compaction.strategies import (
    HierarchicalSummarizationCompactor,
    OffloadToMemoryCompactor,
    SelectiveRetentionCompactor,
    SummarizerFn,
)

__all__ = [
    "Compactor",
    "CompactionConfig",
    "DEFAULT_COMPACTION_THRESHOLD",
    "estimate_state_tokens",
    "message_text",
    "split_keep_tail",
    "SelectiveRetentionCompactor",
    "HierarchicalSummarizationCompactor",
    "OffloadToMemoryCompactor",
    "SummarizerFn",
]
