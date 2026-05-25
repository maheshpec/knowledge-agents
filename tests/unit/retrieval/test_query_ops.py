"""Tests for query operations: Rewriter and Phase-3 stubs (SPEC §7.6.2)."""

import pytest

from common.schemas import Query
from knowledge_index.retrieval.query_ops import (
    Decomposer,
    HyDEExpander,
    Rewriter,
    Stepback,
    apply_query_ops,
)


@pytest.mark.asyncio
async def test_rewriter_appends_rewrite():
    async def fake_complete(prompt: str) -> str:
        return "  refined query  "

    q = Query(raw="what about it?")
    out = await Rewriter(complete=fake_complete).transform(q)
    assert out.rewrites == ["refined query"]
    assert out.raw == "what about it?"  # original preserved


@pytest.mark.asyncio
async def test_rewriter_skips_noop_rewrite():
    async def echo(prompt: str) -> str:
        return "what about it?"  # identical to raw -> no-op

    q = Query(raw="what about it?")
    out = await Rewriter(complete=echo).transform(q)
    assert out.rewrites == []


@pytest.mark.asyncio
async def test_apply_query_ops_threads_through():
    calls = []

    class TagOp:
        name = "tag"

        def __init__(self, tag: str) -> None:
            self.tag = tag

        async def transform(self, query: Query) -> Query:
            calls.append(self.tag)
            return query.model_copy(update={"rewrites": [*query.rewrites, self.tag]})

    out = await apply_query_ops([TagOp("a"), TagOp("b")], Query(raw="x"))
    assert calls == ["a", "b"]  # applied in order
    assert out.rewrites == ["a", "b"]


@pytest.mark.asyncio
async def test_apply_empty_ops_is_identity():
    q = Query(raw="x")
    assert await apply_query_ops([], q) is q


@pytest.mark.asyncio
async def test_query_op_stubs_raise():
    for stub in (HyDEExpander(), Decomposer(), Stepback()):
        with pytest.raises(NotImplementedError):
            await stub.transform(Query(raw="x"))
