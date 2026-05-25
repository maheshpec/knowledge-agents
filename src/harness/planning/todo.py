"""Plan-and-Execute planner (SPEC §6.2) — STUB, lands in Phase 2.

The orchestrator selects ReAct in Phase 1; this class exists so the planner
registry surface is complete and the Phase 2 work has a clear home.
"""

from __future__ import annotations

from typing import Any

from common.schemas import Plan
from harness.planning.base import PlanningContext


class TodoListPlanner:
    """Decompose a goal into a todo list and execute (Plan-and-Execute)."""

    name = "todo_list"

    async def plan(self, goal: str, context: PlanningContext) -> Plan:
        raise NotImplementedError("TodoListPlanner lands in Phase 2 (SPEC §6.2 / §10)")

    async def adapt(self, plan: Plan, new_observation: Any) -> Plan:
        raise NotImplementedError("TodoListPlanner lands in Phase 2 (SPEC §6.2 / §10)")


__all__ = ["TodoListPlanner"]
