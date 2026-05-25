"""Sparse BM25 retriever (SPEC §7.6.3).

BM25 scoring is computed server-side by Qdrant over the collection's sparse
vectors, so this retriever just forwards the query text and lets the index do the
lexical match. No embedding step is required.
"""

from __future__ import annotations

from common.schemas import Query, RetrievalCandidate
from harness.observability.tracing import traced
from knowledge_index.retrieval.retrievers.base import (
    SupportsSearch,
    build_search_filters,
)


class SparseBM25Retriever:
    """Lexical retrieval over Convoy B's ``Index.search_sparse``."""

    name = "sparse_bm25"

    def __init__(self, index: SupportsSearch) -> None:
        self._index = index

    def _query_text(self, query: Query) -> str:
        """Use the raw query for lexical match — rewrites can drop keywords BM25 needs."""
        return query.raw

    @traced(span_name="retrieval.sparse_bm25")
    async def retrieve(self, query: Query, k: int) -> list[RetrievalCandidate]:
        return await self._index.search_sparse(
            self._query_text(query), k, build_search_filters(query)
        )


__all__ = ["SparseBM25Retriever"]
