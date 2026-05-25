"""Permissions / approval gates (SPEC §6.10): pause the graph for sensitive actions."""

from common.types import ApprovalRequest, ApprovalResponse
from harness.permissions.base import (
    ActionType,
    BaseGate,
    Gate,
    PendingAction,
    evaluate_gates,
)
from harness.permissions.gates import (
    DEFAULT_GATE_BUDGET_THRESHOLD,
    DEFAULT_MAX_CONCURRENT_SUBAGENTS,
    BudgetGate,
    ConcurrencyGate,
    WriteGate,
    default_gates,
)
from harness.permissions.graph import GateNode, make_gate_node

__all__ = [
    "ApprovalRequest",
    "ApprovalResponse",
    "ActionType",
    "PendingAction",
    "Gate",
    "BaseGate",
    "evaluate_gates",
    "WriteGate",
    "BudgetGate",
    "ConcurrencyGate",
    "default_gates",
    "DEFAULT_GATE_BUDGET_THRESHOLD",
    "DEFAULT_MAX_CONCURRENT_SUBAGENTS",
    "GateNode",
    "make_gate_node",
]
