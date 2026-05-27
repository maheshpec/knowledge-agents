"""Self-improvement loop (SPEC §8): registry, ledger, budget guard, reviewer.

Phase 4O wires the experiment ledger (§8.2.1) and budget guard (§8.5) that the
Phase 4P evolutionary loop consumes. Re-exported here so the loop can import the
operational primitives from one place.
"""

from __future__ import annotations

from self_improvement.budget_guard import BudgetConfig, BudgetExceeded, BudgetGuard
from self_improvement.ledger import (
    Experiment,
    ExperimentLedger,
    JSONLLedger,
    MutationRecord,
    ReviewerVerdict,
    RunManifest,
)

__all__ = [
    "Experiment",
    "ExperimentLedger",
    "JSONLLedger",
    "MutationRecord",
    "ReviewerVerdict",
    "RunManifest",
    "BudgetConfig",
    "BudgetExceeded",
    "BudgetGuard",
]
