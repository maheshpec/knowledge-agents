"""Planner contract (SPEC §6.2).

Phase 1 ships the ReAct planner (no explicit plan — route → act → observe one
step at a time). Plan-and-Execute (``TodoListPlanner``) is stubbed for Phase 2.
Plans are :class:`common.schemas.Plan`, which already serializes to/from JSON.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from common.schemas import Plan


class PlanningContext(BaseModel):
    """Inputs a planner may use to shape a plan (SPEC §6.2)."""

    budget_remaining: float = 0.0
    max_hops: int = 1
    user_principals: list[str] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Planner(Protocol):
    """Produce and adapt a plan toward a goal (SPEC §6.2)."""

    name: str

    async def plan(self, goal: str, context: PlanningContext) -> Plan: ...
    async def adapt(self, plan: Plan, new_observation: Any) -> Plan: ...


__all__ = ["PlanningContext", "Planner"]
