"""LangGraph ``interrupt`` integration for permission gates (SPEC §6.10).

:func:`make_gate_node` builds a node that evaluates a list of gates against the
current state. If a gate trips, the node calls LangGraph's ``interrupt`` with the
:class:`ApprovalRequest` payload — pausing the graph and checkpointing it. The
chat layer surfaces the request; the user's :class:`ApprovalResponse` is fed back
via ``Command(resume=...)`` and becomes the return value of ``interrupt``.

On approval the node clears ``pending_action`` and records the decision; on denial
it clears the action and flags ``approval_denied`` so the orchestrator can route
around the blocked step instead of executing it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from common.types import ApprovalResponse
from harness.observability.logging import get_logger
from harness.permissions.base import Gate, evaluate_gates

_log = get_logger("harness.permissions")

GateNode = Callable[[Mapping[str, Any]], Awaitable[dict]]


def _coerce_response(value: Any) -> ApprovalResponse:
    """Normalize the resume payload into an :class:`ApprovalResponse`."""
    if isinstance(value, ApprovalResponse):
        return value
    if isinstance(value, Mapping):
        return ApprovalResponse.model_validate(dict(value))
    # A bare truthy/falsey resume is treated as the approval decision.
    return ApprovalResponse(request_id="", approved=bool(value))


def make_gate_node(gates: list[Gate]) -> GateNode:
    """Build an async LangGraph node that pauses on the first tripped gate.

    The node returns a state delta: an empty dict when no gate fires (pass
    through), or the cleared action + decision record after approval/denial.
    """

    async def gate_node(state: Mapping[str, Any]) -> dict:
        request = evaluate_gates(list(gates), state)
        if request is None:
            return {}

        # Imported here so the rest of the module (and unit tests of the gates
        # themselves) need no langgraph runtime present.
        from langgraph.types import interrupt

        _log.info("permissions.pause", gate=request.gate, reason=request.reason)
        resume_value = interrupt(request.model_dump())
        response = _coerce_response(resume_value)

        decision = {
            "request_id": response.request_id or request.request_id,
            "gate": request.gate,
            "approved": response.approved,
            "note": response.note,
        }
        if response.approved:
            _log.info("permissions.approved", gate=request.gate)
            return {"pending_action": None, "approval": decision}
        _log.info("permissions.denied", gate=request.gate)
        return {"pending_action": None, "approval_denied": True, "approval": decision}

    return gate_node


__all__ = ["GateNode", "make_gate_node"]
