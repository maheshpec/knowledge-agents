"""Tests for rerankers: Cohere wrapper, Null baseline, stubs (SPEC §7.6.5)."""

import pytest

from knowledge_index.retrieval.reranking import (
    CohereReranker,
    LLMReranker,
    NullReranker,
    VoyageReranker,
)


@pytest.mark.asyncio
async def test_null_reranker_preserves_order_and_truncates(candidate_fn):
    cands = [candidate_fn(f"c{i}", float(i), "rrf", i) for i in range(5)]
    out = await NullReranker().rerank("q", cands, top_k=3)
    assert [c.chunk.chunk_id for c in out] == ["c0", "c1", "c2"]


@pytest.mark.asyncio
async def test_cohere_reranker_reorders_by_relevance(fake_cohere_client, make_chunk_fn):
    from common.schemas import RetrievalCandidate

    # Fake client ranks by document length; give c_long the longest text.
    cands = [
        RetrievalCandidate(
            chunk=make_chunk_fn("c_short", "hi"), score=0.9, retriever="rrf", rank=1
        ),
        RetrievalCandidate(
            chunk=make_chunk_fn("c_long", "a much longer document body here"),
            score=0.1,
            retriever="rrf",
            rank=2,
        ),
    ]
    reranker = CohereReranker(client=fake_cohere_client)
    out = await reranker.rerank("q", cands, top_k=2)
    assert out[0].chunk.chunk_id == "c_long"
    assert out[0].retriever == "cohere"
    assert [c.rank for c in out] == [1, 2]


@pytest.mark.asyncio
async def test_cohere_reranker_empty_candidates(fake_cohere_client):
    out = await CohereReranker(client=fake_cohere_client).rerank("q", [], top_k=5)
    assert out == []


@pytest.mark.asyncio
async def test_cohere_reranker_uses_context_in_doc_text(make_chunk_fn):
    from knowledge_index.retrieval.reranking.cohere import _doc_text

    chunk = make_chunk_fn("c", "body", context="situating context")
    from common.schemas import RetrievalCandidate

    cand = RetrievalCandidate(chunk=chunk, score=1.0, retriever="rrf", rank=1)
    assert "situating context" in _doc_text(cand)
    assert "body" in _doc_text(cand)


@pytest.mark.asyncio
async def test_reranker_stubs_raise(candidate_fn):
    for stub in (VoyageReranker(), LLMReranker()):
        with pytest.raises(NotImplementedError):
            await stub.rerank("q", [candidate_fn("a", 1.0, "rrf", 1)], top_k=1)
