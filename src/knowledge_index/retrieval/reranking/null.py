"""No-op reranker — the baseline (SPEC §7.6.5).

Preserves fusion order and truncates to ``top_k``. Used as the control arm in
the self-improvement loop: any real reranker must beat this to justify its cost.
"""

from __future__ import annotations

from common.schemas import RetrievalCandidate
from harness.observability.tracing import traced


class NullReranker:
    """Pass-through reranker: keep input order, truncate to top_k."""

    name = "null"

    @traced(span_name="retrieval.rerank.null")
    async def rerank(
        self, query: str, candidates: list[RetrievalCandidate], top_k: int
    ) -> list[RetrievalCandidate]:
        return candidates[:top_k]


__all__ = ["NullReranker"]
