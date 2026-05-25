"""Reranker protocol (SPEC §7.6.5).

A reranker re-scores fused candidates against the query with a model stronger
than first-stage retrieval (typically a cross-encoder) and returns the top-k.
The ``query`` argument is the raw query string — rerankers score text, not
embeddings.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from common.schemas import RetrievalCandidate


@runtime_checkable
class Reranker(Protocol):
    """Re-score and truncate candidates against the query (SPEC §7.6.5)."""

    name: str

    async def rerank(
        self, query: str, candidates: list[RetrievalCandidate], top_k: int
    ) -> list[RetrievalCandidate]: ...


__all__ = ["Reranker"]
