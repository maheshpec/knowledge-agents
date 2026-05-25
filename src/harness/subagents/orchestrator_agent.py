"""Default sub-agent handler: a fresh orchestrator per task (SPEC §6.4).

Each invocation builds its **own** compiled LangGraph orchestrator and runs the
task string as the question — satisfying the "each sub-agent is its own LangGraph
instance, clean context" contract. The child budget is settled with the run's
real cost so it bubbles up to the parent (SPEC §6.4, §6.11).

The returned model is the orchestrator's :class:`GenerationResult`; set
``SubAgentTask.return_schema = GenerationResult`` (or a compatible subset).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from harness.budget.tracker import BudgetTracker
from harness.orchestrator.state import OrchestratorDeps
from harness.subagents.base import AgentFn, SubAgentTask


def orchestrator_agent_fn(deps: OrchestratorDeps, *, k: int = 10, max_hops: int = 1) -> AgentFn:
    """Build an :data:`AgentFn` that runs a fresh orchestrator per task."""

    async def _run(task: SubAgentTask, child: BudgetTracker) -> Any:
        from harness.orchestrator.graph import build_orchestrator, initial_state

        app = build_orchestrator(deps)  # fresh LangGraph instance per sub-agent
        state = initial_state(task.task, budget_usd=child.available(), k=k, max_hops=max_hops)
        cfg = {"configurable": {"thread_id": str(uuid4())}}
        final = await app.ainvoke(state, cfg)
        result = final["result"]
        # Settle the child budget with the real cost — bubbles up to the parent.
        if result.cost > 0:
            amount = min(result.cost, child.available())
            if amount > 0:
                grant = child.reserve(amount)
                child.consume(grant, amount)
        return result

    return _run


__all__ = ["orchestrator_agent_fn"]
