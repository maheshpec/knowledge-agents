"""Retrievers: dense, sparse BM25, and the parallel gather harness (SPEC §7.6.3)."""

from knowledge_index.retrieval.retrievers.base import (
    ACL_FILTER_KEY,
    Retriever,
    SupportsEmbedQuery,
    SupportsSearch,
    build_search_filters,
    gather_retrievers,
)
from knowledge_index.retrieval.retrievers.dense import DenseRetriever
from knowledge_index.retrieval.retrievers.sparse import SparseBM25Retriever

__all__ = [
    "ACL_FILTER_KEY",
    "Retriever",
    "SupportsEmbedQuery",
    "SupportsSearch",
    "build_search_filters",
    "gather_retrievers",
    "DenseRetriever",
    "SparseBM25Retriever",
]
