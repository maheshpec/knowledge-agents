"""Data models for the adversarial reviewer (SPEC §8.3).

The reviewer runs a fixed battery of *checks* against a candidate evaluation and
its lineage; each yields a :class:`CheckResult`. Those fold into a
:class:`ReviewerVerdict` whose ``verdict`` (``accept | reject |
needs_more_evidence``) gates PR creation (SPEC §8.4). Thresholds are collected in
:class:`ReviewThresholds` so the gate is tunable without touching check logic.

The reviewer compares a candidate against its lineage (SPEC §8.2.1). Rather than
depend on the not-yet-built §8.2.1 ``ExperimentLedger``, it consumes a structural
:class:`LineageProvider` yielding :class:`LineageEntry` records — a 4O adapter
maps stored ``Experiment``\\s to these, so the ledger drops in unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from evaluation.runners.runner import EvalReport

# The three outcomes the verdict gate understands (SPEC §8.3).
Verdict = str  # Literal narrowed below via the model field
VERDICTS = ("accept", "reject", "needs_more_evidence")


class CheckResult(BaseModel):
    """The outcome of one adversarial check.

    ``critical`` marks a check whose failure invalidates the result outright
    (forces ``reject``); a non-critical failure only demotes the verdict to
    ``needs_more_evidence`` — the improvement may be real but is unproven.
    """

    name: str
    passed: bool
    critical: bool
    summary: str
    detail: dict[str, object] = Field(default_factory=dict)


class ReviewerVerdict(BaseModel):
    """The reviewer's gate decision plus its structured critique (SPEC §8.3)."""

    verdict: str = Field(pattern="^(accept|reject|needs_more_evidence)$")
    critique: str = ""
    checks: list[CheckResult] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)

    @property
    def gates_pr(self) -> bool:
        """Only an ``accept`` verdict permits PR creation (SPEC §8.4)."""
        return self.verdict == "accept"

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


class ReviewThresholds(BaseModel):
    """Tunable gate thresholds (SPEC §8.3)."""

    # Which aggregated metric is the headline improvement being judged.
    primary_metric: str = "recall@5"
    # Improvement must clear this many sigmas of seed variance to count as real.
    noise_band_sigmas: float = 2.0
    # Fraction of shared queries that must improve for the gain to be "broad".
    min_improved_fraction: float = 0.30
    # Max tolerated relative regression in mean latency / cost vs the baseline.
    latency_regression_pct: float = 0.20
    cost_regression_pct: float = 0.20
    # Per-query deltas smaller than this are treated as noise (ties).
    epsilon: float = 1e-9


class LineageEntry(BaseModel):
    """One ancestor of the candidate, as the reviewer needs to see it.

    ``is_seed`` flags entries produced by a ``seed`` mutation (SPEC §8.2.1): their
    spread defines the noise band. ``report`` is the ancestor's evaluation.
    """

    experiment_id: str
    is_seed: bool
    report: EvalReport

    model_config = {"arbitrary_types_allowed": True}


@runtime_checkable
class LineageProvider(Protocol):
    """The slice of the §8.2.1 ``ExperimentLedger`` the reviewer consumes.

    A Phase 4O adapter implements this by mapping each stored ``Experiment`` in an
    experiment's lineage to a :class:`LineageEntry` (config + eval results +
    seed flag), letting the reviewer compare a candidate against its ancestry
    without taking a hard dependency on the ledger's storage model.
    """

    async def lineage_entries(self, experiment_id: str) -> list[LineageEntry]: ...


__all__ = [
    "Verdict",
    "VERDICTS",
    "CheckResult",
    "ReviewerVerdict",
    "ReviewThresholds",
    "LineageEntry",
    "LineageProvider",
]
