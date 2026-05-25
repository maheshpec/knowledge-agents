"""PostProcessor protocol and vector helpers (SPEC §7.6.6).

Post-processors run after reranking, in a composable list, each transforming the
candidate set (diversify, expand to parents, dedupe, …).
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from common.schemas import Query, RetrievalCandidate


@runtime_checkable
class PostProcessor(Protocol):
    """Transform the candidate set after reranking (SPEC §7.6.6)."""

    name: str

    async def process(
        self, query: Query, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]: ...


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 if either is degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


__all__ = ["PostProcessor", "cosine"]
