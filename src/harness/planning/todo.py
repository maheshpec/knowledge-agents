"""Plan-and-Execute planner (SPEC §6.2) — the Phase 2 todo-list planner.

Unlike ReAct (act one step at a time, no explicit plan), the ``TodoListPlanner``
asks an LLM to decompose a goal into an explicit todo list of
:class:`common.schemas.PlanStep` up front. Steps carry ``depends_on`` links, so
:meth:`execute` resolves them as a DAG: every step whose dependencies are
complete runs concurrently (``asyncio.gather``), then the next wave runs. Plans
serialize to/from JSON via :class:`common.schemas.Plan` for inspection and
persistence; :meth:`adapt` re-plans the unfinished tail when a step fails.

The LLM call is an injected ``CompleteFn`` (``str -> str``) so the planner runs
fully offline under test. :func:`default_completer` supplies a real Haiku-backed
completer in production.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from common.schemas import Plan, PlanStep
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from harness.planning.base import PlanningContext
from knowledge_index.retrieval.query_ops.base import CompleteFn, default_completer

_log = get_logger("harness.planning.todo")

# A step runner executes a single PlanStep and returns its observation/result.
StepRunner = Callable[[PlanStep], Awaitable[Any]]

PLAN_PROMPT = """You are a planning assistant. Decompose the goal into a minimal todo \
list of concrete steps.

Return ONLY a JSON array (no prose, no code fences). Each element is an object:
  {{"id": "<short-id>", "description": "<what to do>", "depends_on": ["<id>", ...]}}

Rules:
- Use depends_on to link a step to the steps that must finish before it.
- Steps with no dependency on each other will run concurrently, so keep
  independent sub-tasks independent.
- Emit at most {max_steps} steps.

Goal: {goal}"""

ADAPT_PROMPT = """A plan step failed while pursuing the goal. Produce a revised todo \
list for the REMAINING work only.

Return ONLY a JSON array of steps (same schema as before):
  {{"id": "<short-id>", "description": "<what to do>", "depends_on": ["<id>", ...]}}

Goal: {goal}
Failed step: {failed}
Observation: {observation}
Already completed (do not repeat): {completed}"""

DEFAULT_MAX_STEPS = 8


def _strip_code_fence(text: str) -> str:
    """Drop a leading/trailing markdown code fence if the LLM wrapped its JSON."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_steps(raw: str) -> list[PlanStep]:
    """Parse an LLM JSON array into validated PlanSteps (tolerant of code fences)."""
    payload = _strip_code_fence(raw)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"planner LLM did not return valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("planner LLM must return a JSON array of steps")

    steps: list[PlanStep] = []
    seen: set[str] = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("each plan step must be a JSON object")
        step_id = str(item.get("id") or f"step-{i + 1}")
        if step_id in seen:
            raise ValueError(f"duplicate step id: {step_id!r}")
        seen.add(step_id)
        steps.append(
            PlanStep(
                id=step_id,
                description=str(item.get("description", "")),
                tool=item.get("tool"),
                inputs=item.get("inputs") or {},
                depends_on=[str(d) for d in item.get("depends_on", [])],
                status="pending",
            )
        )
    # Drop dangling/self dependencies so the DAG is always resolvable.
    for step in steps:
        step.depends_on = [d for d in step.depends_on if d in seen and d != step.id]
    return steps


class TodoListPlanner:
    """Decompose a goal into a todo-list DAG and execute it (Plan-and-Execute)."""

    name = "todo_list"

    def __init__(self, complete: CompleteFn | None = None, *, max_steps: int = DEFAULT_MAX_STEPS):
        self._complete = complete or default_completer()
        self._max_steps = max_steps

    @traced(span_name="planning.todo.plan")
    async def plan(self, goal: str, context: PlanningContext) -> Plan:
        # max_hops scales how many steps are worth planning given the budget.
        cap = min(self._max_steps, max(1, context.max_hops * self._max_steps))
        raw = await self._complete(PLAN_PROMPT.format(goal=goal, max_steps=cap))
        steps = _parse_steps(raw)
        if not steps:
            # Degenerate goal — fall back to a single answer step so the graph runs.
            steps = [PlanStep(id="answer", description=goal, status="pending")]
        return Plan(goal=goal, steps=steps, status="executing")

    @traced(span_name="planning.todo.adapt")
    async def adapt(self, plan: Plan, new_observation: Any) -> Plan:
        failed = self._failed_step(plan, new_observation)
        if failed is None:
            return plan  # nothing went wrong — keep the plan as is

        completed = [s.id for s in plan.steps if s.status == "done"]
        raw = await self._complete(
            ADAPT_PROMPT.format(
                goal=plan.goal,
                failed=failed.id,
                observation=json.dumps(new_observation, default=str),
                completed=json.dumps(completed),
            )
        )
        new_steps = _parse_steps(raw)
        if not new_steps:
            # LLM offered no recovery; mark the plan terminally failed.
            return plan.model_copy(update={"status": "failed"})

        # Preserve completed history, replace the unfinished tail with the new plan.
        kept = [s for s in plan.steps if s.status == "done"]
        return Plan(goal=plan.goal, steps=[*kept, *new_steps], status="executing")

    @staticmethod
    def _failed_step(plan: Plan, observation: Any) -> PlanStep | None:
        """Identify the failed step from the plan state or the observation payload."""
        if isinstance(observation, dict):
            sid = observation.get("failed_step")
            if sid is not None:
                return next((s for s in plan.steps if s.id == sid), None)
            if observation.get("failed"):
                return next(
                    (s for s in plan.steps if s.status in ("running", "pending")),
                    plan.steps[0] if plan.steps else None,
                )
        return next((s for s in plan.steps if s.status == "failed"), None)

    @traced(span_name="planning.todo.execute")
    async def execute(self, plan: Plan, run_step: StepRunner) -> Plan:
        """Run the plan as a DAG: independent steps run concurrently per wave.

        ``run_step`` executes one step and returns its result. A step runs once
        all of its ``depends_on`` predecessors are ``done``. If a dependency
        failed (or was skipped), dependents are marked ``skipped``. Each wave of
        ready steps is dispatched with :func:`asyncio.gather`, so total time
        tracks the critical path, not the sum of all steps.
        """
        by_id = {s.id: s for s in plan.steps}

        def _blocked(step: PlanStep) -> bool:
            return any(by_id[d].status in ("failed", "skipped") for d in step.depends_on)

        def _ready(step: PlanStep) -> bool:
            return step.status == "pending" and all(
                by_id[d].status == "done" for d in step.depends_on
            )

        async def _run_one(step: PlanStep) -> None:
            step.status = "running"
            try:
                step.result = await run_step(step)
                step.status = "done"
            except Exception as exc:  # a failed step shouldn't abort the whole DAG
                step.result = {"error": str(exc)}
                step.status = "failed"
                _log.warning("planning.todo.step_failed", step=step.id, error=str(exc))

        while True:
            # Cascade skips: any pending step whose dependency failed/skipped.
            progressed = False
            for step in plan.steps:
                if step.status == "pending" and _blocked(step):
                    step.status = "skipped"
                    progressed = True

            wave = [s for s in plan.steps if _ready(s)]
            if not wave:
                if progressed:
                    continue  # re-evaluate after cascading skips
                break
            await asyncio.gather(*(_run_one(s) for s in wave))

        if any(s.status == "failed" for s in plan.steps):
            plan.status = "failed"
        elif all(s.status in ("done", "skipped") for s in plan.steps):
            plan.status = "completed"
        return plan


__all__ = ["TodoListPlanner", "StepRunner", "PLAN_PROMPT", "ADAPT_PROMPT"]
