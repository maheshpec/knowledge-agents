"""Data models and collaborator protocols for the evolutionary loop (SPEC §8.2).

The loop's working unit is a :class:`Candidate`: a full :class:`PipelineConfig`
genome plus the lineage, mutation record, eval scores, and reviewer verdict that
accumulate as it flows through a generation (mutate → evaluate → review → select).

The loop depends only on the *protocols* declared here — :class:`Evaluator`,
:class:`Reviewer`, :class:`BudgetGuard`, :class:`ExperimentLedger` — so the Phase
4O ledger/budget-guard and the §8.3 adversarial reviewer can be swapped in without
the loop changing. Tests inject trivial stubs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from self_improvement.registry.pipeline_config import PipelineConfig

Verdict = Literal["accept", "reject", "needs_more_evidence"]


class MutationRecord(BaseModel):
    """What changed relative to a candidate's parent(s) (SPEC §8.2.1)."""

    type: Literal["seed", "mutate", "crossover"]
    component: str = ""  # the gene that changed, e.g. "chunker" or "mmr_lambda"
    change: dict[str, Any] = Field(default_factory=dict)  # before/after diff


def config_hash(config: PipelineConfig) -> str:
    """Content-addressable hash of a config; identical genomes share results."""
    payload = config.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class Candidate(BaseModel):
    """A pipeline genome plus the state accumulated through one generation."""

    candidate_id: str
    config: PipelineConfig
    generation: int = 0
    parent_ids: list[str] = Field(default_factory=list)
    mutation: MutationRecord | None = None

    # Filled by the evaluator (keyed metric name -> value) on the dev set, and
    # optionally on the rotating set for the Goodhart guard (SPEC §8.2).
    metrics: dict[str, float] = Field(default_factory=dict)
    rotating_metrics: dict[str, float] | None = None

    # Filled by the loop (composite) and the reviewer (verdict).
    score: float | None = None
    verdict: Verdict | None = None

    cost_usd: float = 0.0
    compute_seconds: float = 0.0

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)


class EvolutionReport(BaseModel):
    """Outcome of an evolutionary run (SPEC §8.2)."""

    run_id: str
    generations_run: int
    stopped_reason: Literal["completed", "budget_exhausted"]
    baseline_score: float
    # Best candidate that cleared the delta threshold + Goodhart guard + reviewer
    # accept verdict; ``None`` if nothing qualified (no PR-worthy improvement).
    best: Candidate | None = None
    best_overall: Candidate | None = None
    population: list[Candidate] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)


@runtime_checkable
class Evaluator(Protocol):
    """Scores candidates on the held-out eval set, filling ``metrics`` (SPEC §9)."""

    async def evaluate_batch(self, candidates: list[Candidate]) -> list[Candidate]: ...


@runtime_checkable
class Reviewer(Protocol):
    """Adversarial review; fills ``verdict`` to gate qualification (SPEC §8.3)."""

    async def review_batch(self, candidates: list[Candidate]) -> list[Candidate]: ...


@runtime_checkable
class BudgetGuard(Protocol):
    """Hard ceilings on generations / compute / cost (SPEC §8.5)."""

    def exhausted(self) -> bool: ...
    def charge(self, *, cost_usd: float = 0.0, compute_seconds: float = 0.0) -> None: ...
    def tick_generation(self) -> None: ...


@runtime_checkable
class ExperimentLedger(Protocol):
    """Append-only, git-backed experiment record store (SPEC §8.2.1).

    The loop calls :meth:`append` with each evaluated candidate; the Phase 4O
    ledger adapter maps a :class:`Candidate` onto its richer ``Experiment`` record.
    """

    async def append(self, record: Any) -> None: ...


__all__ = [
    "Verdict",
    "MutationRecord",
    "config_hash",
    "Candidate",
    "EvolutionReport",
    "Evaluator",
    "Reviewer",
    "BudgetGuard",
    "ExperimentLedger",
]
