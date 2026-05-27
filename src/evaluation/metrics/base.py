"""Metric contract (SPEC §9.2).

A metric scores one query at a time (``compute``) and folds the per-query
results into a single dataset-level number (``aggregate``). Splitting the two
lets the runner compute metrics in parallel per query, then reduce once. Every
metric is identified by a stable ``name`` so reports and LangSmith experiments
are comparable across runs.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from common.schemas import GenerationResult, GoldQuery, RetrievalCandidate


class QueryOutcome(BaseModel):
    """Everything a metric may inspect for a single query (SPEC §9.2/§9.3).

    Carries the gold label, the ranked retrieval candidates, the final generated
    answer (when end-to-end), and operational telemetry (latency/cost/tokens).
    """

    gold: GoldQuery
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    generation: GenerationResult | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    # tokens that ended up cited / used vs. the full prompt (token efficiency)
    useful_tokens: int = 0
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class MetricResult(BaseModel):
    """A single metric's value for one query (or, after aggregate, the dataset)."""

    name: str
    value: float
    detail: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Metric(Protocol):
    """Protocol every metric implements (SPEC §9.2)."""

    name: str

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        """Score a single query."""
        ...

    def aggregate(self, results: list[MetricResult]) -> MetricResult:
        """Fold per-query results into one dataset-level result."""
        ...


def mean_aggregate(name: str, results: list[MetricResult]) -> MetricResult:
    """Default aggregation: arithmetic mean over the per-query values."""
    values = [r.value for r in results if r.value == r.value]  # drop NaN
    value = sum(values) / len(values) if values else 0.0
    return MetricResult(name=name, value=value, detail={"n": len(values)})


class MeanMetric:
    """Base for metrics whose dataset value is the mean of per-query values."""

    name: str = "mean_metric"

    def compute(self, outcome: QueryOutcome) -> MetricResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def aggregate(self, results: list[MetricResult]) -> MetricResult:
        return mean_aggregate(self.name, results)


__all__ = [
    "QueryOutcome",
    "MetricResult",
    "Metric",
    "MeanMetric",
    "mean_aggregate",
]
