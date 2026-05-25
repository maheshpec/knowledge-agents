"""Permission gate contracts (SPEC §6.10).

A *gate* inspects the orchestrator state — specifically the ``pending_action`` it
is about to take — and decides whether a human must approve before proceeding.
Gates are pure predicates (``should_pause``) plus a request builder, so they are
trivially testable in isolation; the LangGraph ``interrupt`` wiring lives in
:mod:`harness.permissions.graph`.

State convention: the node about to act sets ``state["pending_action"]`` to a
:class:`PendingAction`-shaped mapping describing what it wants to do. Gates read
it; an empty/absent pending action never pauses.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from common.types import ApprovalRequest

ActionType = Literal["external_write", "tool", "spawn"]


class PendingAction(TypedDict, total=False):
    """The action a node intends to take, surfaced for gate inspection."""

    type: ActionType
    description: str
    cost: float  # estimated USD for a tool/LLM action
    spawn_count: int  # sub-agents this action would start


@runtime_checkable
class Gate(Protocol):
    """A permission gate: pause the graph when an action needs approval (SPEC §6.10)."""

    name: str

    def should_pause(self, state: Mapping[str, Any]) -> bool: ...

    def build_request(self, state: Mapping[str, Any]) -> ApprovalRequest: ...


class BaseGate:
    """Shared plumbing: pull the pending action and build a standard request."""

    name: str = "base"
    risk: Literal["low", "medium", "high"] = "medium"

    def should_pause(self, state: Mapping[str, Any]) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def pending(state: Mapping[str, Any]) -> Mapping[str, Any]:
        action = state.get("pending_action")
        return action if isinstance(action, Mapping) else {}

    def _reason(self, state: Mapping[str, Any]) -> str:
        return f"{self.name} gate tripped"

    def build_request(self, state: Mapping[str, Any]) -> ApprovalRequest:
        action = self.pending(state)
        return ApprovalRequest(
            gate=self.name,
            action=str(action.get("description", action.get("type", "unknown action"))),
            reason=self._reason(state),
            risk=self.risk,
            payload=dict(action),
        )


def evaluate_gates(gates: list[Gate], state: Mapping[str, Any]) -> ApprovalRequest | None:
    """Return an approval request for the first gate that wants to pause, else None."""
    for gate in gates:
        if gate.should_pause(state):
            return gate.build_request(state)
    return None


__all__ = [
    "ActionType",
    "PendingAction",
    "Gate",
    "BaseGate",
    "evaluate_gates",
]
