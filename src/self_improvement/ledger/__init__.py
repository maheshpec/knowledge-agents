"""Experiment ledger (SPEC §8.2.1): versioned, replayable, git-backed JSONL.

The evolutionary loop (§8.2 / Phase 4P) appends an :class:`Experiment` per
candidate, updates status as it is evaluated/reviewed, and can :meth:`replay`
any experiment from its stored config to verify reproducibility.
"""

from __future__ import annotations

from self_improvement.ledger.ids import config_hash, uuid7
from self_improvement.ledger.ledger import (
    Evaluator,
    ExperimentLedger,
    JSONLLedger,
    load_jsonl,
)
from self_improvement.ledger.models import (
    Experiment,
    ExperimentStatus,
    MutationRecord,
    ReviewerVerdict,
    RunManifest,
)

__all__ = [
    "uuid7",
    "config_hash",
    "Experiment",
    "ExperimentStatus",
    "MutationRecord",
    "ReviewerVerdict",
    "RunManifest",
    "ExperimentLedger",
    "JSONLLedger",
    "Evaluator",
    "load_jsonl",
]
