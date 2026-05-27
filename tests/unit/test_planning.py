"""Tests for the ReAct planner (SPEC §6.2).

The Phase 2G ``TodoListPlanner`` is covered in ``test_planner.py``.
"""

from common.schemas import Plan
from harness.planning import PlanningContext, ReactPlanner


async def test_react_planner_returns_executing_single_step():
    plan = await ReactPlanner().plan("answer the question", PlanningContext(max_hops=1))
    assert plan.status == "executing"
    assert len(plan.steps) == 1
    assert plan.steps[0].status == "pending"


async def test_react_plan_serializes_roundtrip():
    plan = await ReactPlanner().plan("g", PlanningContext())
    dumped = plan.model_dump_json()
    assert Plan.model_validate_json(dumped).goal == "g"


async def test_react_adapt_marks_failed_step_on_failure():
    planner = ReactPlanner()
    plan = await planner.plan("g", PlanningContext())
    adapted = await planner.adapt(plan, {"failed": True, "candidates": 0})
    assert adapted.steps[0].status == "failed"


async def test_react_adapt_keeps_step_on_success():
    planner = ReactPlanner()
    plan = await planner.plan("g", PlanningContext())
    adapted = await planner.adapt(plan, {"failed": False, "candidates": 5})
    assert adapted.steps[0].status != "failed"
