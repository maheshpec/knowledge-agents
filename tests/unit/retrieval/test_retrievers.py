"""Tests for dense/sparse retrievers, ACL filtering, and the parallel harness (SPEC §7.6.3)."""

import asyncio
import time

import pytest

from common.schemas import Query, RetrievalCandidate
from knowledge_index.retrieval.retrievers import (
    DenseRetriever,
    SparseBM25Retriever,
    build_search_filters,
    gather_retrievers,
)
from knowledge_index.retrieval.retrievers.base import ACL_FILTER_KEY


def _corpus(make_chunk_fn):
    return [
        make_chunk_fn("c1", "alpha beta gamma", acl=["team-a"]),
        make_chunk_fn("c2", "beta delta", acl=[]),  # public
        make_chunk_fn("c3", "secret epsilon", acl=["team-b"]),
    ]


def test_build_search_filters_injects_principals():
    q = Query(raw="hi", user_principals=["u1", "u2"], filters={"lang": "en"})
    filters = build_search_filters(q)
    assert filters[ACL_FILTER_KEY] == ["u1", "u2"]
    assert filters["lang"] == "en"


@pytest.mark.asyncio
async def test_dense_retriever_returns_candidates(fake_index_cls, fake_embedder_cls, make_chunk_fn):
    index = fake_index_cls(_corpus(make_chunk_fn))
    retriever = DenseRetriever(index, fake_embedder_cls())
    q = Query(raw="beta", user_principals=["team-a"])
    out = await retriever.retrieve(q, k=10)
    assert all(isinstance(c, RetrievalCandidate) for c in out)
    assert out and out[0].retriever == "dense"


@pytest.mark.asyncio
async def test_sparse_retriever_ranks_by_term_overlap(fake_index_cls, make_chunk_fn):
    index = fake_index_cls(_corpus(make_chunk_fn))
    retriever = SparseBM25Retriever(index)
    q = Query(raw="beta", user_principals=["team-a", "team-b"])
    out = await retriever.retrieve(q, k=10)
    ids = [c.chunk.chunk_id for c in out]
    # c1 and c2 both contain "beta"; c3 does not — it ranks last (score 0).
    assert ids[-1] == "c3"


@pytest.mark.asyncio
async def test_acl_mismatch_returns_zero_hits(fake_index_cls, make_chunk_fn):
    # A principal matching nothing private; only the public chunk c2 is visible.
    index = fake_index_cls(
        [
            make_chunk_fn("p1", "private one", acl=["team-a"]),
            make_chunk_fn("p2", "private two", acl=["team-b"]),
        ]
    )
    retriever = SparseBM25Retriever(index)
    q = Query(raw="private", user_principals=["intruder"])
    out = await retriever.retrieve(q, k=10)
    assert out == []


@pytest.mark.asyncio
async def test_gather_runs_retrievers_concurrently(fake_index_cls, make_chunk_fn):
    class SlowRetriever:
        name = "slow"

        def __init__(self, delay: float) -> None:
            self.delay = delay

        async def retrieve(self, query, k):
            await asyncio.sleep(self.delay)
            return []

    retrievers = [SlowRetriever(0.1), SlowRetriever(0.1)]
    start = time.perf_counter()
    await gather_retrievers(retrievers, Query(raw="x"), k=5)
    elapsed = time.perf_counter() - start
    # Concurrent: ~0.1s total, not 0.2s. Generous bound for CI jitter.
    assert elapsed < 0.18


@pytest.mark.asyncio
async def test_gather_empty_returns_empty():
    assert await gather_retrievers([], Query(raw="x"), k=5) == []
