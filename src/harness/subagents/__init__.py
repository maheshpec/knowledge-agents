"""Sub-agents (SPEC §6.4): clean-context delegation with budget bubbling."""

from __future__ import annotations

from harness.subagents.base import AgentFn, SubAgentResult, SubAgentTask
from harness.subagents.orchestrator_agent import orchestrator_agent_fn
from harness.subagents.runner import spawn, spawn_all

__all__ = [
    "SubAgentTask",
    "SubAgentResult",
    "AgentFn",
    "spawn",
    "spawn_all",
    "orchestrator_agent_fn",
]
