"""Index contract + sparse encoders (SPEC §7.5, §3 Sparse note).

The index stores chunks as hybrid points (dense vector + sparse BM25-style
vector) and enforces ACLs at search time via payload filters. Sparse encoding
is pluggable: ``FastEmbedBM25Encoder`` is the production path (Qdrant/bm25), and
``HashingBM25Encoder`` is a dependency-free fallback for offline dev/tests and
the rank_bm25 sidecar bake-off described in SPEC §3.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from common.schemas import Chunk, RetrievalCandidate

SparseVector = tuple[list[int], list[float]]  # (indices, values)


class SnapshotRef(BaseModel):
    """A handle to a point-in-time index snapshot (SPEC §7.5 versioning)."""

    collection: str
    name: str
    backend: str = "local"  # "local" (json) | "qdrant" (native)
    location: str | None = None
    count: int = 0


@runtime_checkable
class Index(Protocol):
    """Hybrid vector store contract (SPEC §7.5)."""

    async def upsert(self, chunks: list[Chunk]) -> None: ...
    async def delete(self, chunk_ids: list[str]) -> None: ...
    async def search_dense(
        self, vec: list[float], k: int, filters: dict
    ) -> list[RetrievalCandidate]: ...
    async def search_sparse(
        self, query: str, k: int, filters: dict
    ) -> list[RetrievalCandidate]: ...
    async def snapshot(self) -> SnapshotRef: ...
    async def restore(self, ref: SnapshotRef) -> None: ...


@runtime_checkable
class SparseEncoder(Protocol):
    """Encode text into sparse (indices, values) vectors."""

    name: str

    def encode_documents(self, texts: list[str]) -> list[SparseVector]: ...
    def encode_query(self, text: str) -> SparseVector: ...


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingBM25Encoder:
    """Dependency-free BM25-style sparse encoder (offline fallback).

    Term frequencies are hashed into a fixed-size vocabulary with a BM25
    saturation transform. No corpus IDF (single-pass, no global stats) — this is
    the recall-limited fallback the SPEC flags for the bake-off, not the primary.
    """

    name = "hashing_bm25"

    def __init__(self, vocab_size: int = 2**18, k1: float = 1.5, b: float = 0.75) -> None:
        self.vocab_size = vocab_size
        self.k1 = k1
        self.b = b
        self._avg_len = 50.0  # static prior; no corpus pass

    def _hash(self, token: str) -> int:
        import hashlib

        return int.from_bytes(
            hashlib.blake2b(token.encode(), digest_size=4).digest(), "big"
        ) % self.vocab_size

    def _encode(self, text: str) -> SparseVector:
        tokens = _tokenize(text)
        if not tokens:
            return ([], [])
        counts = Counter(self._hash(t) for t in tokens)
        doc_len = len(tokens)
        indices: list[int] = []
        values: list[float] = []
        for idx, tf in counts.items():
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_len)
            score = tf * (self.k1 + 1) / denom
            indices.append(idx)
            values.append(round(score, 6))
        return (indices, values)

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._encode(t) for t in texts]

    def encode_query(self, text: str) -> SparseVector:
        # Binary-weight the query terms (presence), standard BM25 query side.
        tokens = set(_tokenize(text))
        if not tokens:
            return ([], [])
        idxs = sorted({self._hash(t) for t in tokens})
        return (idxs, [1.0] * len(idxs))


class FastEmbedBM25Encoder:
    """Production sparse encoder using fastembed's Qdrant/bm25 (lazy import)."""

    name = "fastembed_bm25"

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from fastembed import SparseTextEmbedding  # type: ignore

            self._model = SparseTextEmbedding(model_name=self.model_name)
        return self._model

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        model = self._get_model()
        return [
            (emb.indices.tolist(), emb.values.tolist())
            for emb in model.embed(texts)
        ]

    def encode_query(self, text: str) -> SparseVector:
        model = self._get_model()
        emb = next(iter(model.query_embed(text)))
        return (emb.indices.tolist(), emb.values.tolist())


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity helper (used by the in-memory test index/MMR)."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


__all__ = [
    "SparseVector",
    "SnapshotRef",
    "Index",
    "SparseEncoder",
    "HashingBM25Encoder",
    "FastEmbedBM25Encoder",
    "cosine",
]
