"""Span extraction post-processor stub (SPEC §7.6.6).

Registered in ``configs/components.yaml`` so the registry/search space can name
it, but the extraction logic (find the exact relevant sub-span within a chunk)
lands in a later phase. Instantiable now; raises if actually run.
"""

from __future__ import annotations

from common.schemas import Query, RetrievalCandidate


class SpanExtractor:
    """Find the exact relevant span within each chunk. Implemented in a later phase."""

    name = "span_extractor"

    async def process(
        self, query: Query, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        raise NotImplementedError("SpanExtractor is a later-phase stub (SPEC §7.6.6)")


__all__ = ["SpanExtractor"]
