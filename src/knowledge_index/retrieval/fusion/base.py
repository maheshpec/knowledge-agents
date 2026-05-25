"""Fuser protocol and shared helpers (SPEC §7.6.4).

A fuser merges several ranked candidate lists (one per retriever) into a single
ranked list, deduplicating chunks that surface in more than one list.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from common.schemas import RetrievalCandidate


@runtime_checkable
class Fuser(Protocol):
    """Combine per-retriever ranked lists into one fused ranking (SPEC §7.6.4)."""

    name: str

    async def fuse(self, results: list[list[RetrievalCandidate]]) -> list[RetrievalCandidate]: ...


def candidate_key(candidate: RetrievalCandidate) -> str:
    """Identity used to merge the same chunk across retriever lists."""
    return candidate.chunk.chunk_id


__all__ = ["Fuser", "candidate_key"]
