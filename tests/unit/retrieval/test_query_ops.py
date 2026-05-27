"""Tests for query operations: Rewriter, HyDE, decompose, step-back (SPEC §7.6.2)."""

import pytest

from common.schemas import Query
from knowledge_index.retrieval.query_ops import (
    Decomposer,
    HyDEExpander,
    Rewriter,
    Stepback,
    apply_query_ops,
)


def _const(text: str):
    """A completer that ignores its prompt and always returns ``text``."""

    async def _complete(prompt: str) -> str:
        return text

    return _complete


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
async def test_hyde_appends_hypothetical_document():
    op = HyDEExpander(complete=_const("  A hypothetical answer passage.  "))
    out = await op.transform(Query(raw="why is the sky blue?"))
    assert out.hyde == ["A hypothetical answer passage."]
    assert out.raw == "why is the sky blue?"  # original preserved


@pytest.mark.asyncio
async def test_hyde_skips_duplicate():
    op = HyDEExpander(complete=_const("dup"))
    q = Query(raw="q", hyde=["dup"])
    out = await op.transform(q)
    assert out.hyde == ["dup"]  # not appended twice


@pytest.mark.asyncio
async def test_decomposer_splits_into_sub_queries():
    op = Decomposer(complete=_const("1. What is X?\n2. What is Y?\n- How do X and Y relate?"))
    out = await op.transform(Query(raw="compare X and Y"))
    assert out.sub_queries == ["What is X?", "What is Y?", "How do X and Y relate?"]


@pytest.mark.asyncio
async def test_decomposer_noop_when_atomic():
    # A single line identical to the raw query means it was already atomic.
    op = Decomposer(complete=_const("how tall is Everest?"))
    out = await op.transform(Query(raw="how tall is Everest?"))
    assert out.sub_queries == []


@pytest.mark.asyncio
async def test_decomposer_dedupes_against_existing():
    op = Decomposer(complete=_const("What is X?\nWhat is Y?"))
    q = Query(raw="q", sub_queries=["What is X?"])
    out = await op.transform(q)
    assert out.sub_queries == ["What is X?", "What is Y?"]


@pytest.mark.asyncio
async def test_stepback_appends_broader_query():
    op = Stepback(complete=_const("What governs the motion of planets?"))
    out = await op.transform(Query(raw="why is Mars's orbit elliptical?"))
    assert out.rewrites == ["What governs the motion of planets?"]


@pytest.mark.asyncio
async def test_stepback_skips_noop():
    op = Stepback(complete=_const("same"))
    out = await op.transform(Query(raw="same"))
    assert out.rewrites == []


@pytest.mark.asyncio
async def test_expanders_compose_into_distinct_fields():
    ops = [
        HyDEExpander(complete=_const("hypo doc")),
        Decomposer(complete=_const("part one?\npart two?")),
        Stepback(complete=_const("broader?")),
    ]
    out = await apply_query_ops(ops, Query(raw="original multi-part question"))
    assert out.hyde == ["hypo doc"]
    assert out.sub_queries == ["part one?", "part two?"]
    assert out.rewrites == ["broader?"]
    assert out.raw == "original multi-part question"
