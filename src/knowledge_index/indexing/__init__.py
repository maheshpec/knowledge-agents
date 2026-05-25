"""Indexing package (SPEC §7.5): hybrid Qdrant store + sparse encoders."""

from __future__ import annotations

from knowledge_index.indexing.base import (
    FastEmbedBM25Encoder,
    HashingBM25Encoder,
    Index,
    SnapshotRef,
    SparseEncoder,
    SparseVector,
    cosine,
)
from knowledge_index.indexing.qdrant_index import DENSE, SPARSE, QdrantIndex

__all__ = [
    "Index",
    "SnapshotRef",
    "SparseEncoder",
    "SparseVector",
    "HashingBM25Encoder",
    "FastEmbedBM25Encoder",
    "cosine",
    "QdrantIndex",
    "DENSE",
    "SPARSE",
]
