"""Evidence-package contracts for self-improvement PRs (SPEC §8.4).

The PR generator turns an *accepted candidate* into an auditable evidence
package: eval metrics before/after, candidate lineage, the adversarial reviewer's
report, and links to the LangSmith trace and held-out test results.

The evolutionary loop (§8.2), experiment ledger (§8.2.1), and adversarial
reviewer (§8.3) are separate Phase-4 modules not present on this branch. Rather
than import them, ``pr_gen`` depends on the *structural* contracts below, shaped
to match the SPEC schemas so the real producers drop in unchanged at integration
time — the same decoupling the retrieval layer uses for Convoy B (SPEC §7.6.3).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from evaluation.metrics.base import MetricResult
from self_improvement.registry.pipeline_config import PipelineConfig

# The reviewer's gate verdict (SPEC §8.3). Only ``accept`` may produce a PR.
Verdict = Literal["accept", "reject", "needs_more_evidence"]


class ReviewerReport(BaseModel):
    """The adversarial reviewer's structured verdict + critique (SPEC §8.3)."""

    verdict: Verdict
    critique: str = ""
    # Named validity checks the reviewer ran -> passed?, e.g. {"leakage_free": True}.
    checks: dict[str, bool] = Field(default_factory=dict)


class LineageEntry(BaseModel):
    """One ancestor of the candidate, distilled from a ledger ``Experiment`` (§8.2.1)."""

    experiment_id: str
    generation: int
    mutation_summary: str = ""  # human-readable "what changed vs parent"
    config_hash: str = ""


class MetricDelta(BaseModel):
    """Before/after movement of a single metric, with absolute + relative change."""

    name: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def pct_change(self) -> float | None:
        """Relative change vs baseline; ``None`` when the baseline is exactly 0."""
        if self.before == 0:
            return None
        return (self.after - self.before) / abs(self.before)

    @property
    def improved(self) -> bool:
        return self.after > self.before


class AcceptedCandidate(BaseModel):
    """The full input the PR generator consumes for one accepted candidate (§8.4).

    Bundles the candidate config, its baseline, the before/after eval results
    (keyed by metric name, as the evaluator emits them), lineage from the ledger,
    the reviewer report, and the evidence links. Everything the PR needs to be
    self-justifying lives here — the generator performs no further evaluation.
    """

    experiment_id: str
    run_id: str
    config: PipelineConfig
    baseline_config: PipelineConfig | None = None
    # Keyed by metric name; ``before`` is the baseline arm, ``after`` the candidate.
    before: dict[str, MetricResult] = Field(default_factory=dict)
    after: dict[str, MetricResult] = Field(default_factory=dict)
    lineage: list[LineageEntry] = Field(default_factory=list)
    reviewer: ReviewerReport
    langsmith_trace_url: str | None = None
    heldout_results_url: str | None = None


class EvidencePackage(BaseModel):
    """The assembled, render-ready evidence for a PR (SPEC §8.4)."""

    experiment_id: str
    run_id: str
    config: PipelineConfig
    baseline_config: PipelineConfig | None
    metric_deltas: list[MetricDelta]
    lineage: list[LineageEntry]
    reviewer: ReviewerReport
    langsmith_trace_url: str | None
    heldout_results_url: str | None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def improved_metrics(self) -> list[MetricDelta]:
        return [d for d in self.metric_deltas if d.improved]

    def regressed_metrics(self) -> list[MetricDelta]:
        return [d for d in self.metric_deltas if d.after < d.before]


def compute_metric_deltas(
    before: dict[str, MetricResult], after: dict[str, MetricResult]
) -> list[MetricDelta]:
    """Pair before/after metrics by name into deltas (union of keys, sorted).

    A metric present on only one side is reported with the missing side at 0.0,
    so a newly-introduced or dropped metric is still visible in the package.
    """
    names = sorted(set(before) | set(after))
    deltas: list[MetricDelta] = []
    for name in names:
        b = before[name].value if name in before else 0.0
        a = after[name].value if name in after else 0.0
        deltas.append(MetricDelta(name=name, before=b, after=a))
    return deltas


def build_evidence_package(candidate: AcceptedCandidate) -> EvidencePackage:
    """Assemble the evidence package from an accepted candidate (no I/O, no eval)."""
    return EvidencePackage(
        experiment_id=candidate.experiment_id,
        run_id=candidate.run_id,
        config=candidate.config,
        baseline_config=candidate.baseline_config,
        metric_deltas=compute_metric_deltas(candidate.before, candidate.after),
        lineage=candidate.lineage,
        reviewer=candidate.reviewer,
        langsmith_trace_url=candidate.langsmith_trace_url,
        heldout_results_url=candidate.heldout_results_url,
    )


__all__ = [
    "Verdict",
    "ReviewerReport",
    "LineageEntry",
    "MetricDelta",
    "AcceptedCandidate",
    "EvidencePackage",
    "compute_metric_deltas",
    "build_evidence_package",
]
