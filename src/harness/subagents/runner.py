"""Sub-agent spawning + budget bubbling (SPEC §6.4).

``spawn`` carves a child budget from the parent (so the child's spend is capped
and bubbles back up), runs the handler with a clean context, and packages a
structured :class:`SubAgentResult`. ``spawn_all`` carves every child budget up
front (deterministic) then runs them concurrently via ``asyncio``.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from harness.budget.tracker import BudgetTracker
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from harness.subagents.base import AgentFn, SubAgentResult, SubAgentTask

_log = get_logger("harness.subagents")

# Floor so we never divide by zero when the parent is fully committed.
_EPS = 1e-9


def _carve_child(parent: BudgetTracker, task: SubAgentTask) -> BudgetTracker:
    """Reserve the task's budget from the parent and return the child tracker.

    The child ceiling is ``min(task.budget.max_usd, parent.available())`` — a
    sub-agent can never be granted more than the parent currently has free, and
    its spend bubbles back into the parent on settle (SPEC §6.4, §6.11).
    """
    available = parent.available()
    requested = task.budget.max_usd
    fraction = 1.0 if requested >= available else max(_EPS, requested / max(available, _EPS))
    return parent.child_budget(fraction)


async def _run(task: SubAgentTask, child: BudgetTracker, agent_fn: AgentFn) -> SubAgentResult:
    trace_id = str(uuid4())
    try:
        result = await agent_fn(task, child)
        if not isinstance(result, task.return_schema):
            # Coerce/validate into the requested schema (fail loudly if it can't).
            result = task.return_schema.model_validate(result.model_dump())
        return SubAgentResult(
            task=task.task,
            ok=True,
            output=result.model_dump(),
            trace_id=trace_id,
            cost=child.consumed,
        )
    except Exception as exc:  # noqa: BLE001 - sub-agent failure is data, not a crash
        _log.warning("subagent.failed", task=task.task, error=str(exc), trace_id=trace_id)
        return SubAgentResult(
            task=task.task, ok=False, error=str(exc), trace_id=trace_id, cost=child.consumed
        )


@traced(span_name="subagents.spawn")
async def spawn(
    task: SubAgentTask, parent_budget: BudgetTracker, agent_fn: AgentFn
) -> SubAgentResult:
    """Spawn a single sub-agent with a clean context and a carved child budget."""
    child = _carve_child(parent_budget, task)
    return await _run(task, child, agent_fn)


@traced(span_name="subagents.spawn_all")
async def spawn_all(
    tasks: list[SubAgentTask], parent_budget: BudgetTracker, agent_fn: AgentFn
) -> list[SubAgentResult]:
    """Spawn many sub-agents in parallel (SPEC §6.4).

    Child budgets are carved sequentially first (so allocation is deterministic
    and the parent can't be over-committed by a race), then the handlers run
    concurrently via ``asyncio.create_task``/``gather``.
    """
    children = [_carve_child(parent_budget, t) for t in tasks]
    coros = [_run(t, c, agent_fn) for t, c in zip(tasks, children, strict=True)]
    aio_tasks = [asyncio.create_task(c) for c in coros]
    return list(await asyncio.gather(*aio_tasks))


__all__ = ["spawn", "spawn_all"]
