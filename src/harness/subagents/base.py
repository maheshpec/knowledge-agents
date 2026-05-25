"""Sub-agent contracts (SPEC §6.4).

A sub-agent runs with a *clean context*: it receives only a :class:`SubAgentTask`
(the task string, the return schema the parent wants, a tool subset, and a
budget) — never the parent's message history. It returns a structured
:class:`SubAgentResult` plus a trace pointer for debugging.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.types import BudgetSpec
from harness.budget.tracker import BudgetTracker


class SubAgentTask(BaseModel):
    """A unit of delegated work handed to a sub-agent (SPEC §6.4)."""

    # return_schema is a pydantic *class*, not an instance — allow arbitrary types.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str
    return_schema: type[BaseModel]  # the shape the parent wants back
    tools: list[str] = Field(default_factory=list)  # subset available to the sub-agent
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    max_turns: int = 20


class SubAgentResult(BaseModel):
    """The structured result returned to the parent (SPEC §6.4)."""

    task: str
    ok: bool
    output: dict[str, Any] | None = None  # conforms to the task's return_schema when ok
    error: str | None = None
    trace_id: str
    cost: float = 0.0


# A sub-agent handler does the work: given the task and its own budget tracker, it
# returns an instance of the task's return_schema. It must not be passed parent
# history — the runner enforces clean context by only handing it these two args.
AgentFn = Callable[[SubAgentTask, BudgetTracker], Awaitable[BaseModel]]

__all__ = ["SubAgentTask", "SubAgentResult", "AgentFn"]
