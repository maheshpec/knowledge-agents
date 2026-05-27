"""Operational metrics (SPEC §9.2): latency p50/p95, cost/query, token efficiency.

These read the runner's telemetry (``QueryOutcome.latency_ms`` etc.) rather than
the answer content. The latency metrics carry the raw per-query value through
``compute`` and reduce with a percentile in ``aggregate``; the others mean-reduce.
"""

from __future__ import annotations

from evaluation.metrics.base import MeanMetric, MetricResult, QueryOutcome


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (``pct`` in [0, 100]); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    # nearest-rank: rank = ceil(pct/100 * n), 1-based, clamped to [1, n]
    rank = max(1, min(len(ordered), -(-int(pct) * len(ordered) // 100)))
    return ordered[rank - 1]


class _LatencyPercentile(MeanMetric):
    def __init__(self, pct: float, name: str) -> None:
        self.pct = pct
        self.name = name

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        return MetricResult(name=self.name, value=outcome.latency_ms)

    def aggregate(self, results: list[MetricResult]) -> MetricResult:
        value = _percentile([r.value for r in results], self.pct)
        return MetricResult(name=self.name, value=value, detail={"n": len(results)})


class LatencyP50(_LatencyPercentile):
    """Median per-query latency in milliseconds."""

    def __init__(self) -> None:
        super().__init__(50.0, "latency_p50")


class LatencyP95(_LatencyPercentile):
    """95th-percentile per-query latency in milliseconds."""

    def __init__(self) -> None:
        super().__init__(95.0, "latency_p95")


class CostPerQuery(MeanMetric):
    """Mean USD cost per query."""

    name = "cost_per_query"

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        return MetricResult(name=self.name, value=outcome.cost_usd)


class TokenEfficiency(MeanMetric):
    """Mean ratio of useful (cited/used) tokens to total tokens consumed."""

    name = "token_efficiency"

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        total = outcome.tokens_in + outcome.tokens_out
        if total <= 0:
            return MetricResult(name=self.name, value=0.0, detail={"total_tokens": 0})
        value = min(1.0, outcome.useful_tokens / total)
        return MetricResult(
            name=self.name,
            value=value,
            detail={"useful": outcome.useful_tokens, "total": total},
        )


def default_operational_metrics() -> list[MeanMetric]:
    """The operational metric suite from ``configs/eval.yaml``."""
    return [LatencyP50(), LatencyP95(), CostPerQuery(), TokenEfficiency()]


__all__ = [
    "LatencyP50",
    "LatencyP95",
    "CostPerQuery",
    "TokenEfficiency",
    "default_operational_metrics",
]
