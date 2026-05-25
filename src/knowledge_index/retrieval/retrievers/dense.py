"""Dense vector retriever (SPEC §7.6.3).

Embeds the query (the rewritten/HyDE-augmented form when present) and asks the
index for the nearest dense neighbours. ACL principals are pushed into the filter
so the store enforces access control.
"""

from __future__ import annotations

from common.schemas import Query, RetrievalCandidate
from harness.observability.tracing import traced
from knowledge_index.retrieval.retrievers.base import (
    SupportsEmbedQuery,
    SupportsSearch,
    build_search_filters,
)


class DenseRetriever:
    """Single-vector dense retrieval over Convoy B's ``Index.search_dense``."""

    name = "dense"

    def __init__(self, index: SupportsSearch, embedder: SupportsEmbedQuery) -> None:
        self._index = index
        self._embedder = embedder

    def _query_text(self, query: Query) -> str:
        """Pick the text to embed: first rewrite if any, else the raw query.

        Query ops (Rewriter/HyDE) populate ``rewrites``/``hyde``; a rewrite is a
        cleaner search target than the raw conversational query when present.
        """
        if query.rewrites:
            return query.rewrites[0]
        return query.raw

    @traced(span_name="retrieval.dense")
    async def retrieve(self, query: Query, k: int) -> list[RetrievalCandidate]:
        vec = await self._embedder.embed_query(self._query_text(query))
        return await self._index.search_dense(vec, k, build_search_filters(query))


__all__ = ["DenseRetriever"]
