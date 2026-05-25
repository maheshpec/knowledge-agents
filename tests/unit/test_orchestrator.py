"""Tests for the orchestrator graph (SPEC §6.1)."""

from uuid import uuid4

from common.schemas import Chunk, Query, RetrievalCandidate, RetrievalResult
from harness.citation import CitationEnforcer, CitedDraft, CitedSegment
from harness.context import DefaultPacker
from harness.orchestrator import OrchestratorDeps, build_orchestrator, initial_state
from harness.orchestrator.graph import async_sqlite_saver
from harness.planning import ReactPlanner


class FakePipeline:
    """Returns preset candidates and counts how many times it was called."""

    def __init__(self, candidates: list[RetrievalCandidate], cost: float = 0.0) -> None:
        self._candidates = candidates
        self.cost = cost
        self.calls = 0

    async def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(
            candidates=self._candidates[:k], query=query, trace_id=uuid4(), cost=self.cost
        )


def _cands(n: int) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            chunk=Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=f"fact number {i}"),
            score=1.0 - i * 0.1,
            retriever="dense",
            rank=i,
        )
        for i in range(n)
    ]


async def _draft_top(question, candidates):
    if not candidates:
        return CitedDraft(refused=True, refusal_reason="no evidence found")
    top = candidates[0].chunk
    return CitedDraft(segments=[CitedSegment(text=top.text, citation_ids=[top.chunk_id])])


def _deps(pipeline) -> OrchestratorDeps:
    return OrchestratorDeps(
        pipeline=pipeline,
        enforcer=CitationEnforcer(draft_fn=_draft_top),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
    )


async def _run(deps, **state_kwargs):
    app = build_orchestrator(deps)
    state = initial_state("what is fact 0?", **state_kwargs)
    cfg = {"configurable": {"thread_id": str(uuid4())}}
    return await app.ainvoke(state, cfg)


async def test_single_retrieval_produces_cited_answer():
    pipe = FakePipeline(_cands(3))
    final = await _run(_deps(pipe), budget_usd=1.0, max_hops=1, k=5)
    assert pipe.calls == 1
    result = final["result"]
    assert result.text == "fact number 0"
    assert len(result.citations) == 1
    # citation precision: every cited id was actually retrieved
    retrieved_ids = {c.chunk.chunk_id for c in final["candidates"]}
    assert all(cit.source.chunk_id in retrieved_ids for cit in result.citations)


async def test_zero_retrieval_when_max_hops_zero():
    pipe = FakePipeline(_cands(3))
    final = await _run(_deps(pipe), budget_usd=1.0, max_hops=0, k=5)
    assert pipe.calls == 0
    assert final["result"].text == "no evidence found"


async def test_two_retrieval_hops():
    pipe = FakePipeline(_cands(3))
    final = await _run(_deps(pipe), budget_usd=1.0, max_hops=2, k=5)
    assert pipe.calls == 2
    assert final["hops"] == 2


async def test_budget_exhausted_finalizes_early():
    pipe = FakePipeline(_cands(3))
    final = await _run(_deps(pipe), budget_usd=0.0, max_hops=1, k=5)
    assert pipe.calls == 0  # never retrieves; routes straight to answer
    assert "budget" in final["result"].text.lower()
    assert final["result"].citations == []


async def test_runs_with_async_sqlite_checkpointer():
    # SPEC §6.1: the graph must run under the (async) SqliteSaver checkpointer.
    pipe = FakePipeline(_cands(3))
    app = build_orchestrator(_deps(pipe), checkpointer=async_sqlite_saver(":memory:"))
    thread_id = str(uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    final = await app.ainvoke(initial_state("what is fact 0?", budget_usd=1.0), cfg)
    assert final["result"].text == "fact number 0"
    # state was checkpointed under the thread id and is retrievable
    snapshot = await app.aget_state(cfg)
    assert snapshot.values["result"].text == "fact number 0"
