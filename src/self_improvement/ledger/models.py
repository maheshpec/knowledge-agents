"""Experiment ledger data models (SPEC §8.2.1).

An :class:`Experiment` is the atomic, replayable record the evolutionary loop
emits per candidate. :class:`MutationRecord` captures what changed vs. the
parent(s); :class:`ReviewerVerdict` is the adversarial reviewer's call (§8.3,
defined here as a light placeholder until that phase fleshes it out).
:class:`RunManifest` is the per-run header stored as ``manifest.yaml``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from evaluation.metrics.base import MetricResult
from self_improvement.ledger.ids import config_hash, uuid7
from self_improvement.registry.pipeline_config import PipelineConfig

ExperimentStatus = Literal[
    "pending", "running", "evaluated", "reviewed", "accepted", "rejected", "failed"
]


def _now() -> datetime:
    return datetime.now(UTC)


class MutationRecord(BaseModel):
    """What changed relative to the parent experiment(s) (SPEC §8.2.1)."""

    type: Literal["mutate", "crossover", "seed"]
    component: str  # e.g. "chunker", "reranker"
    change: dict[str, Any] = Field(default_factory=dict)  # before/after diff


class ReviewerVerdict(BaseModel):
    """Adversarial reviewer outcome for an experiment (SPEC §8.3 placeholder)."""

    decision: Literal["accept", "reject", "revise"]
    rationale: str = ""
    confidence: float = 0.0
    concerns: list[str] = Field(default_factory=list)


class Experiment(BaseModel):
    """One atomic, replayable evolutionary experiment record (SPEC §8.2.1)."""

    experiment_id: str = Field(default_factory=lambda: str(uuid7()))
    parent_ids: list[str] = Field(default_factory=list)
    generation: int = 0
    run_id: str
    config: PipelineConfig
    config_hash: str = ""
    mutation: MutationRecord | None = None
    status: ExperimentStatus = "pending"
    eval_results: dict[str, MetricResult] | None = None
    reviewer_verdict: ReviewerVerdict | None = None
    cost_usd: float = 0.0
    compute_seconds: float = 0.0
    created_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None
    trace_ids: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, _ctx: Any) -> None:
        # Derive the content address once on construction if not supplied.
        if not self.config_hash:
            object.__setattr__(self, "config_hash", config_hash(self.config))


class RunManifest(BaseModel):
    """Per-run header (``runs/{run_id}/manifest.yaml``) — SPEC §8.2.1 layout."""

    run_id: str
    generations: int = 0
    population_size: int = 0
    dataset_refs: list[str] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
    notes: str = ""


__all__ = [
    "ExperimentStatus",
    "MutationRecord",
    "ReviewerVerdict",
    "Experiment",
    "RunManifest",
]
