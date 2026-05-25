"""End-to-end HybridPipeline composition tests (SPEC §7.6).

Wires the full graph against the in-memory fakes and asserts the acceptance
criteria from epic ka-2ap: k reranked candidates out, ACL honored, and the
pipeline composes query-ops → retrievers → fusion → rerank → post-proc.
"""

import pytest

from common.schemas import Query, RetrievalResult
from knowledge_index.retrieval import (
    DenseRetriever,
    HybridPipeline,
    NullReranker,
    RRFFuser,
    SparseBM25Retriever,
    build_default_pipeline,
)
from knowledge_index.retrieval.post import DeduplicatorPostProcessor
from knowledge_index.retrieval.reranking import CohereReranker


def _public_corpus(make_chunk_fn, n: int):
    # Every chunk contains "topic" so both retrievers surface them all.
    return [make_chunk_fn(f"c{i}", f"topic content number {i}", acl=[]) for i in range(n)]


@pytest.mark.asyncio
async def test_pipeline_returns_k_reranked_candidates(
    fake_index_cls, fake_embedder_cls, make_chunk_fn
):
    index = fake_index_cls(_public_corpus(make_chunk_fn, 15))
    pipeline = HybridPipeline(
        retrievers=[DenseRetriever(index, fake_embedder_cls()), SparseBM25Retriever(index)],
        reranker=NullReranker(),
        fuser=RRFFuser(),
        post_processors=[DeduplicatorPostProcessor()],
    )
    result = await pipeline.retrieve(Query(raw="topic"), k=10)
    assert isinstance(result, RetrievalResult)
    assert len(result.candidates) == 10
    assert result.latency_ms >= 0.0
    assert result.trace_id is not None
    # ranks are contiguous 1..10
    assert [c.rank for c in result.candidates] == list(range(1, 11))


@pytest.mark.asyncio
async def test_pipeline_honors_acl_zero_hits(fake_index_cls, fake_embedder_cls, make_chunk_fn):
    private = [make_chunk_fn(f"p{i}", f"topic {i}", acl=["team-a"]) for i in range(12)]
    index = fake_index_cls(private)
    pipeline = HybridPipeline(
        retrievers=[DenseRetriever(index, fake_embedder_cls()), SparseBM25Retriever(index)],
        reranker=NullReranker(),
    )
    result = await pipeline.retrieve(Query(raw="topic", user_principals=["intruder"]), k=10)
    assert result.candidates == []


@pytest.mark.asyncio
async def test_pipeline_with_cohere_reranker(
    fake_index_cls, fake_embedder_cls, fake_cohere_client, make_chunk_fn
):
    index = fake_index_cls(_public_corpus(make_chunk_fn, 8))
    pipeline = HybridPipeline(
        retrievers=[DenseRetriever(index, fake_embedder_cls()), SparseBM25Retriever(index)],
        reranker=CohereReranker(client=fake_cohere_client),
    )
    result = await pipeline.retrieve(Query(raw="topic"), k=5)
    assert len(result.candidates) == 5
    assert all(c.retriever == "cohere" for c in result.candidates)


@pytest.mark.asyncio
async def test_build_default_pipeline_wires_canonical_graph(
    fake_index_cls, fake_embedder_cls, fake_cohere_client, make_chunk_fn
):
    index = fake_index_cls(_public_corpus(make_chunk_fn, 10))

    async def fetch_parent(_pid):
        return None

    pipeline = build_default_pipeline(
        index,
        fake_embedder_cls(),
        fetch_parent=fetch_parent,
        cohere_client=fake_cohere_client,
    )
    result = await pipeline.retrieve(Query(raw="topic"), k=5)
    assert len(result.candidates) == 5
    # Cohere reranked, then MMR re-ranked the survivors -> contiguous ranks.
    assert [c.rank for c in result.candidates] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_pipeline_requires_a_retriever():
    with pytest.raises(ValueError):
        HybridPipeline(retrievers=[], reranker=NullReranker())


@pytest.mark.asyncio
async def test_pipeline_rejects_nonpositive_k(fake_index_cls, fake_embedder_cls, make_chunk_fn):
    index = fake_index_cls(_public_corpus(make_chunk_fn, 3))
    pipeline = HybridPipeline(retrievers=[SparseBM25Retriever(index)], reranker=NullReranker())
    with pytest.raises(ValueError):
        await pipeline.retrieve(Query(raw="topic"), k=0)
