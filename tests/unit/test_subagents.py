"""Tests for sub-agent delegation (SPEC §6.4)."""

import asyncio

from pydantic import BaseModel

from common.errors import BudgetExceeded
from common.types import BudgetSpec
from harness.budget.tracker import BudgetTracker
from harness.subagents import SubAgentTask, spawn, spawn_all


class Answer(BaseModel):
    value: str


def _task(text: str, max_usd: float = 30.0) -> SubAgentTask:
    return SubAgentTask(task=text, return_schema=Answer, budget=BudgetSpec(max_usd=max_usd))


async def test_subagent_returns_structured_result():
    async def agent_fn(task, child):
        return Answer(value=task.task.upper())

    result = await spawn(_task("hello"), BudgetTracker(100.0), agent_fn)
    assert result.ok
    assert result.output == {"value": "HELLO"}
    assert result.trace_id  # trace pointer for debugging


async def test_subagent_clean_context_receives_only_task():
    seen: dict = {}

    async def agent_fn(task, child):
        # The handler is handed only the task + its budget — no parent history.
        seen["args"] = (task.task, type(child).__name__)
        return Answer(value="ok")

    await spawn(_task("just the task"), BudgetTracker(50.0), agent_fn)
    assert seen["args"] == ("just the task", "BudgetTracker")


async def test_parallel_spawn_runs_concurrently():
    started = 0
    max_concurrent = 0

    async def agent_fn(task, child):
        nonlocal started, max_concurrent
        started += 1
        max_concurrent = max(max_concurrent, started)
        await asyncio.sleep(0.02)  # hold so siblings overlap
        started -= 1
        return Answer(value=task.task)

    tasks = [_task(f"t{i}") for i in range(3)]
    results = await spawn_all(tasks, BudgetTracker(100.0), agent_fn)
    assert all(r.ok for r in results)
    assert max_concurrent > 1  # they actually overlapped


async def test_budget_bubbles_up_to_parent():
    async def agent_fn(task, child):
        grant = child.reserve(5.0)
        child.consume(grant, 5.0)
        return Answer(value="spent")

    parent = BudgetTracker(100.0)
    await spawn_all([_task(f"t{i}") for i in range(3)], parent, agent_fn)
    assert parent.consumed == 15.0  # 3 children × $5 bubbled up


async def test_child_budget_capped_at_grant():
    # parent budget = 100; child gets 30; child cannot exceed 30.
    captured: dict = {}

    async def agent_fn(task, child):
        captured["limit"] = child.limit
        # attempting to reserve beyond the child ceiling must fail
        try:
            child.reserve(40.0)
            captured["over_reserve_allowed"] = True
        except BudgetExceeded:
            captured["over_reserve_allowed"] = False
        return Answer(value="ok")

    result = await spawn(_task("capped", max_usd=30.0), BudgetTracker(100.0), agent_fn)
    assert result.ok
    assert captured["limit"] == 30.0
    assert captured["over_reserve_allowed"] is False


async def test_subagent_failure_is_captured_not_raised():
    async def agent_fn(task, child):
        raise RuntimeError("boom")

    result = await spawn(_task("explode"), BudgetTracker(100.0), agent_fn)
    assert result.ok is False
    assert "boom" in result.error


async def test_orchestrator_agent_fn_runs_fresh_graph_per_task():
    # The default handler runs each sub-agent as its own LangGraph orchestrator.
    from uuid import uuid4

    from common.schemas import Chunk, GenerationResult, Query, RetrievalCandidate, RetrievalResult
    from harness.citation import CitationEnforcer, CitedDraft, CitedSegment
    from harness.context import DefaultPacker
    from harness.orchestrator import OrchestratorDeps
    from harness.planning import ReactPlanner
    from harness.subagents import orchestrator_agent_fn

    class _Pipe:
        cost = 0.0

        async def retrieve(self, query: Query, k: int) -> RetrievalResult:
            cand = RetrievalCandidate(
                chunk=Chunk(chunk_id="x1", doc_id="d", text="delegated answer"),
                score=1.0,
                retriever="dense",
                rank=1,
            )
            return RetrievalResult(candidates=[cand], query=query, trace_id=uuid4())

    async def draft_fn(question, candidates):
        top = candidates[0].chunk
        return CitedDraft(segments=[CitedSegment(text=top.text, citation_ids=[top.chunk_id])])

    deps = OrchestratorDeps(
        pipeline=_Pipe(),
        enforcer=CitationEnforcer(draft_fn=draft_fn),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
    )
    agent_fn = orchestrator_agent_fn(deps)
    tasks = [
        SubAgentTask(task=f"sub-question {i}", return_schema=GenerationResult) for i in range(2)
    ]
    results = await spawn_all(tasks, BudgetTracker(10.0), agent_fn)
    assert all(r.ok for r in results)
    assert all(r.output["text"] == "delegated answer" for r in results)
