"""Default permission gates (SPEC §6.10).

Three gates cover the sensitive actions the orchestrator can take:

- :class:`WriteGate` — any write to an external system (send email, write a file
  outside the sandbox);
- :class:`BudgetGate` — a single action whose estimated cost exceeds a threshold;
- :class:`ConcurrencyGate` — spawning sub-agents beyond the concurrency cap.

Thresholds default to the values in ``configs/default.yaml`` (SPEC §6.10) but are
constructor args so tests and the registry can vary them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from harness.permissions.base import BaseGate

DEFAULT_GATE_BUDGET_THRESHOLD = 0.25  # USD, mirrors configs/default.yaml
DEFAULT_MAX_CONCURRENT_SUBAGENTS = 3


class WriteGate(BaseGate):
    """Pause before any write to an external system."""

    name = "write"
    risk = "high"

    def should_pause(self, state: Mapping[str, Any]) -> bool:
        return self.pending(state).get("type") == "external_write"

    def _reason(self, state: Mapping[str, Any]) -> str:
        return "Action writes to an external system and needs approval"


class BudgetGate(BaseGate):
    """Pause when a single action's estimated cost exceeds the threshold."""

    name = "budget"
    risk = "medium"

    def __init__(self, threshold: float = DEFAULT_GATE_BUDGET_THRESHOLD) -> None:
        self.threshold = threshold

    def should_pause(self, state: Mapping[str, Any]) -> bool:
        return float(self.pending(state).get("cost", 0.0)) > self.threshold

    def _reason(self, state: Mapping[str, Any]) -> str:
        cost = float(self.pending(state).get("cost", 0.0))
        return f"Action cost ${cost:.2f} exceeds gate threshold ${self.threshold:.2f}"


class ConcurrencyGate(BaseGate):
    """Pause when spawning sub-agents beyond the concurrency cap."""

    name = "concurrency"
    risk = "medium"

    def __init__(self, max_concurrent: int = DEFAULT_MAX_CONCURRENT_SUBAGENTS) -> None:
        self.max_concurrent = max_concurrent

    def should_pause(self, state: Mapping[str, Any]) -> bool:
        action = self.pending(state)
        if action.get("type") != "spawn":
            return False
        active = int(state.get("active_subagents", 0))
        requested = int(action.get("spawn_count", 1))
        return active + requested > self.max_concurrent

    def _reason(self, state: Mapping[str, Any]) -> str:
        active = int(state.get("active_subagents", 0))
        requested = int(self.pending(state).get("spawn_count", 1))
        return (
            f"Spawning {requested} sub-agent(s) with {active} active exceeds "
            f"max_concurrent={self.max_concurrent}"
        )


def default_gates(
    *,
    budget_threshold: float = DEFAULT_GATE_BUDGET_THRESHOLD,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT_SUBAGENTS,
) -> list[BaseGate]:
    """The standard gate set, ordered most-sensitive first."""
    return [WriteGate(), BudgetGate(budget_threshold), ConcurrencyGate(max_concurrent)]


__all__ = [
    "DEFAULT_GATE_BUDGET_THRESHOLD",
    "DEFAULT_MAX_CONCURRENT_SUBAGENTS",
    "WriteGate",
    "BudgetGate",
    "ConcurrencyGate",
    "default_gates",
]
