"""ReAct planner (SPEC §6.2) — the Phase 1 default.

ReAct keeps no explicit multi-step plan: the orchestrator routes, acts, and
observes one step at a time. ``plan`` therefore returns a single-step Plan that
simply marks the loop as executing; ``adapt`` records the latest observation and
flags a failed step so the orchestrator can decide whether to re-route.
"""

from __future__ import annotations

from typing import Any

from common.schemas import Plan, PlanStep
from harness.planning.base import PlanningContext


class ReactPlanner:
    """Null-plan, act-one-step-at-a-time planner (SPEC §6.2)."""

    name = "react"

    async def plan(self, goal: str, context: PlanningContext) -> Plan:
        step = PlanStep(
            id="act",
            description="Retrieve supporting evidence, then answer with citations.",
            status="pending",
        )
        return Plan(goal=goal, steps=[step], status="executing")

    async def adapt(self, plan: Plan, new_observation: Any) -> Plan:
        # ReAct doesn't rewrite the plan; it records the observation and, if the
        # observation signals failure, marks the active step so the router can
        # choose to retry/re-route rather than proceed.
        if plan.steps:
            active = plan.steps[0]
            active.result = new_observation
            if isinstance(new_observation, dict) and new_observation.get("failed"):
                active.status = "failed"
        return plan


__all__ = ["ReactPlanner"]
