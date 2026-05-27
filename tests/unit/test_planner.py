"""Tests for the Plan-and-Execute TodoListPlanner (SPEC §6.2, Phase 2G).

Covers plan serialization, DAG parallel execution, concurrency timing
(3 independent sub-tasks ≈ slowest step, not the sum), and adapt-on-failure.
All LLM calls use an injected fake completer, so these run fully offline.
"""

import asyncio
import json
import time

import pytest

from common.schemas import Plan, PlanStep
from harness.planning import PlanningContext, TodoListPlanner


def _completer(reply: str):
    """A CompleteFn that ignores the prompt and returns canned JSON."""

    async def _complete(prompt: str) -> str:
        return reply

    return _complete


def _scripted_completer(replies: list[str]):
    """A CompleteFn that returns successive canned replies (for plan→adapt)."""
    it = iter(replies)

    async def _complete(prompt: str) -> str:
        return next(it)

    return _complete


THREE_INDEPENDENT = json.dumps(
    [
        {"id": "a", "description": "task a", "depends_on": []},
        {"id": "b", "description": "task b", "depends_on": []},
        {"id": "c", "description": "task c", "depends_on": []},
    ]
)

DIAMOND = json.dumps(
    [
        {"id": "root", "description": "fetch", "depends_on": []},
        {"id": "left", "description": "analyze left", "depends_on": ["root"]},
        {"id": "right", "description": "analyze right", "depends_on": ["root"]},
        {"id": "join", "description": "combine", "depends_on": ["left", "right"]},
    ]
)


async def test_plan_parses_steps_from_llm():
    planner = TodoListPlanner(_completer(THREE_INDEPENDENT))
    plan = await planner.plan("do three things", PlanningContext(max_hops=1))
    assert plan.status == "executing"
    assert [s.id for s in plan.steps] == ["a", "b", "c"]
    assert all(s.status == "pending" for s in plan.steps)


async def test_plan_serializes_roundtrip():
    planner = TodoListPlanner(_completer(DIAMOND))
    plan = await planner.plan("g", PlanningContext())
    restored = Plan.model_validate_json(plan.model_dump_json())
    assert restored.goal == "g"
    assert restored.steps[-1].depends_on == ["left", "right"]


async def test_plan_drops_dangling_and_self_dependencies():
    raw = json.dumps(
        [
            {"id": "x", "description": "x", "depends_on": ["x", "ghost"]},
            {"id": "y", "description": "y", "depends_on": ["x"]},
        ]
    )
    plan = await TodoListPlanner(_completer(raw)).plan("g", PlanningContext())
    assert plan.steps[0].depends_on == []  # self + ghost removed
    assert plan.steps[1].depends_on == ["x"]


async def test_plan_rejects_invalid_json():
    with pytest.raises(ValueError):
        await TodoListPlanner(_completer("not json")).plan("g", PlanningContext())


async def test_empty_plan_falls_back_to_single_step():
    plan = await TodoListPlanner(_completer("[]")).plan("just answer", PlanningContext())
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "just answer"


async def test_execute_respects_dependencies():
    planner = TodoListPlanner(_completer(DIAMOND))
    plan = await planner.plan("g", PlanningContext())
    order: list[str] = []

    async def run_step(step: PlanStep):
        order.append(step.id)
        return f"ran {step.id}"

    done = await planner.execute(plan, run_step)
    assert done.status == "completed"
    assert all(s.status == "done" for s in done.steps)
    # root before its dependents; join last.
    assert order[0] == "root"
    assert order[-1] == "join"
    assert order.index("left") < order.index("join")
    assert order.index("right") < order.index("join")


async def test_execute_runs_independent_steps_concurrently():
    """3 independent sub-tasks should run in ≈ one step's time, not the sum."""
    planner = TodoListPlanner(_completer(THREE_INDEPENDENT))
    plan = await planner.plan("g", PlanningContext())
    delay = 0.1

    async def run_step(step: PlanStep):
        await asyncio.sleep(delay)
        return step.id

    start = time.perf_counter()
    done = await planner.execute(plan, run_step)
    elapsed = time.perf_counter() - start

    assert done.status == "completed"
    # Concurrent: ≈ one delay. Sequential would be ≈ 3×delay. Generous bound.
    assert elapsed < delay * 2.5


async def test_execute_skips_dependents_of_failed_step():
    planner = TodoListPlanner(_completer(DIAMOND))
    plan = await planner.plan("g", PlanningContext())

    async def run_step(step: PlanStep):
        if step.id == "left":
            raise RuntimeError("boom")
        return step.id

    done = await planner.execute(plan, run_step)
    by_id = {s.id: s for s in done.steps}
    assert by_id["left"].status == "failed"
    assert by_id["right"].status == "done"
    assert by_id["join"].status == "skipped"  # depends on the failed step
    assert done.status == "failed"


async def test_adapt_replans_remaining_work_on_failure():
    recovery = json.dumps([{"id": "retry", "description": "retry it", "depends_on": []}])
    planner = TodoListPlanner(_scripted_completer([THREE_INDEPENDENT, recovery]))
    plan = await planner.plan("g", PlanningContext())
    plan.steps[0].status = "done"  # 'a' completed
    plan.steps[1].status = "failed"  # 'b' failed

    adapted = await planner.adapt(plan, {"failed_step": "b"})
    assert adapted.status == "executing"
    ids = [s.id for s in adapted.steps]
    assert "a" in ids  # completed step preserved
    assert "retry" in ids  # recovery step added
    assert "b" not in ids  # failed/pending tail replaced


async def test_adapt_noop_when_nothing_failed():
    planner = TodoListPlanner(_completer(THREE_INDEPENDENT))
    plan = await planner.plan("g", PlanningContext())
    adapted = await planner.adapt(plan, {"failed": False, "candidates": 5})
    assert adapted is plan  # unchanged, no LLM re-plan
