"""Evaluation framework (SPEC §9): datasets, metrics, runners."""

from __future__ import annotations

from evaluation.datasets import Dataset, evolution_mode, load_dataset
from evaluation.metrics import MetricResult, QueryOutcome, default_metrics
from evaluation.runners import EvalReport, EvalRunner, OrchestratorPipelineRunner

__all__ = [
    "Dataset",
    "load_dataset",
    "evolution_mode",
    "default_metrics",
    "MetricResult",
    "QueryOutcome",
    "EvalRunner",
    "EvalReport",
    "OrchestratorPipelineRunner",
]
