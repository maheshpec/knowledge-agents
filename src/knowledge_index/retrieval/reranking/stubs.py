"""Reranker stubs filled in later phases (SPEC §7.6.5).

Registered now so the component registry and config can reference them, but they
raise ``NotImplementedError`` until their phase lands. Keeping the names live
avoids churn in configs/components.yaml when the implementations arrive.
"""

from __future__ import annotations

from common.schemas import RetrievalCandidate


class VoyageReranker:
    """Voyage rerank-2 cross-encoder. Implemented in a later phase."""

    name = "voyage"

    async def rerank(
        self, query: str, candidates: list[RetrievalCandidate], top_k: int
    ) -> list[RetrievalCandidate]:
        raise NotImplementedError("VoyageReranker is a Phase-later stub (SPEC §7.6.5)")


class LLMReranker:
    """LLM listwise reranker — slower, occasionally better. Later phase."""

    name = "llm"

    async def rerank(
        self, query: str, candidates: list[RetrievalCandidate], top_k: int
    ) -> list[RetrievalCandidate]:
        raise NotImplementedError("LLMReranker is a Phase-later stub (SPEC §7.6.5)")


__all__ = ["VoyageReranker", "LLMReranker"]
