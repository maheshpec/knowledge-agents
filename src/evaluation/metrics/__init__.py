"""Evaluation metrics (SPEC §9.2): retrieval, end-to-end, operational."""

from __future__ import annotations

from evaluation.metrics.base import (
    MeanMetric,
    Metric,
    MetricResult,
    QueryOutcome,
    mean_aggregate,
)
from evaluation.metrics.e2e import (
    AnswerRelevance,
    AnswerScorer,
    CitationJudge,
    CitationPrecision,
    CitationRecall,
    Faithfulness,
    LexicalOverlapJudge,
    default_e2e_metrics,
)
from evaluation.metrics.operational import (
    CostPerQuery,
    LatencyP50,
    LatencyP95,
    TokenEfficiency,
    default_operational_metrics,
)
from evaluation.metrics.retrieval import (
    MRR,
    HitRate,
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
    default_retrieval_metrics,
)


def default_metrics(judge: CitationJudge | None = None) -> list[MeanMetric]:
    """The full metric suite (retrieval + end-to-end + operational)."""
    return [
        *default_retrieval_metrics(),
        *default_e2e_metrics(judge),
        *default_operational_metrics(),
    ]


__all__ = [
    # base
    "Metric",
    "MeanMetric",
    "MetricResult",
    "QueryOutcome",
    "mean_aggregate",
    # retrieval
    "RecallAtK",
    "PrecisionAtK",
    "NDCGAtK",
    "MRR",
    "HitRate",
    "default_retrieval_metrics",
    # e2e
    "CitationJudge",
    "LexicalOverlapJudge",
    "CitationPrecision",
    "CitationRecall",
    "AnswerScorer",
    "Faithfulness",
    "AnswerRelevance",
    "default_e2e_metrics",
    # operational
    "LatencyP50",
    "LatencyP95",
    "CostPerQuery",
    "TokenEfficiency",
    "default_operational_metrics",
    # combined
    "default_metrics",
]
