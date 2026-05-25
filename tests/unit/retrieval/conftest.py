"""Offline fakes for the retrieval pipeline tests (Phase 1C).

Convoy B's QdrantIndex/Embedder are not on this branch, so these in-memory fakes
stand in. ``FakeIndex`` enforces ACL the way the real index must (payload-filter
intersection), and ``FakeEmbedder`` returns fixed vectors so cosine-based tests
(MMR) are deterministic.
"""

from __future__ import annotations

import pytest

from common.schemas import Chunk, RetrievalCandidate
from knowledge_index.retrieval.retrievers.base import ACL_FILTER_KEY


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    parent_id: str | None = None,
    acl: list[str] | None = None,
    context: str | None = None,
    embedding: list[float] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        parent_id=parent_id,
        text=text,
        context=context,
        embedding=embedding,
        acl=acl or [],
    )


def _acl_visible(chunk: Chunk, principals: list[str]) -> bool:
    """Public (empty acl) is visible to all; otherwise principals must intersect."""
    if not chunk.acl:
        return True
    return bool(set(chunk.acl) & set(principals))


class FakeEmbedder:
    """Deterministic embedder: looks up fixed vectors, falls back to a hash vector."""

    name = "fake"
    dim = 3

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self._vectors = vectors or {}

    async def embed_query(self, text: str) -> list[float]:
        if text in self._vectors:
            return self._vectors[text]
        h = abs(hash(text))
        return [float((h >> (8 * i)) & 0xFF) for i in range(self.dim)]


class FakeIndex:
    """In-memory index honoring ACL filters, scoring by lexical overlap.

    ``search_dense`` ignores the vector and ranks by token overlap with the query
    text recorded at construction-time per chunk; good enough to exercise the
    pipeline wiring. ``search_sparse`` ranks by raw substring match count.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def _filter_acl(self, filters: dict) -> list[Chunk]:
        principals = filters.get(ACL_FILTER_KEY, [])
        return [c for c in self._chunks if _acl_visible(c, principals)]

    async def search_dense(self, vec, k, filters):
        visible = self._filter_acl(filters)

        # Score by vector L1 closeness when chunk has an embedding, else by order.
        def score(c: Chunk) -> float:
            if c.embedding and len(c.embedding) == len(vec):
                return -sum(abs(a - b) for a, b in zip(c.embedding, vec, strict=True))
            return 0.0

        ranked = sorted(visible, key=score, reverse=True)[:k]
        return [
            RetrievalCandidate(chunk=c, score=score(c), retriever="dense", rank=i)
            for i, c in enumerate(ranked, start=1)
        ]

    async def search_sparse(self, query, k, filters):
        visible = self._filter_acl(filters)
        terms = query.lower().split()

        def score(c: Chunk) -> int:
            return sum(c.text.lower().count(t) for t in terms)

        ranked = sorted(visible, key=score, reverse=True)[:k]
        return [
            RetrievalCandidate(chunk=c, score=float(score(c)), retriever="sparse_bm25", rank=i)
            for i, c in enumerate(ranked, start=1)
        ]


class FakeCohereResult:
    def __init__(self, index: int, relevance_score: float) -> None:
        self.index = index
        self.relevance_score = relevance_score


class FakeCohereResponse:
    def __init__(self, results: list[FakeCohereResult]) -> None:
        self.results = results


class FakeCohereClient:
    """Reranks by descending document length (deterministic, no network)."""

    async def rerank(self, *, model, query, documents, top_n):
        order = sorted(range(len(documents)), key=lambda i: len(documents[i]), reverse=True)
        results = [
            FakeCohereResult(index=i, relevance_score=1.0 - rank * 0.1)
            for rank, i in enumerate(order[:top_n])
        ]
        return FakeCohereResponse(results)


def candidate(chunk_id: str, score: float, retriever: str, rank: int) -> RetrievalCandidate:
    """Build a bare candidate for fusion/rerank/post unit tests."""
    return RetrievalCandidate(
        chunk=make_chunk(chunk_id, f"text for {chunk_id}"),
        score=score,
        retriever=retriever,
        rank=rank,
    )


# --- fixtures (helpers exposed so test modules avoid tests-package import) ---


@pytest.fixture
def make_chunk_fn():
    return make_chunk


@pytest.fixture
def candidate_fn():
    return candidate


@pytest.fixture
def fake_embedder_cls():
    return FakeEmbedder


@pytest.fixture
def fake_index_cls():
    return FakeIndex


@pytest.fixture
def fake_cohere_client():
    return FakeCohereClient()
