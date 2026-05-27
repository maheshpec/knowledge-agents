"""Evolutionary self-improvement loop (SPEC §8.2): mutate, cross, evaluate, select."""

from self_improvement.evolutionary.budget import SimpleBudgetGuard
from self_improvement.evolutionary.genome import (
    FUSION_CHOICES,
    RERANKER_TOP_K_RANGE,
    RETRIEVER_POOL,
    crossover,
    mutate,
    random_config,
)
from self_improvement.evolutionary.loop import EvolutionaryLoop
from self_improvement.evolutionary.selection import (
    DEFAULT_DELTA_THRESHOLD,
    DEFAULT_PRIMARY_METRIC,
    ScorePolicy,
    composite_score,
    qualifies,
    select,
)
from self_improvement.evolutionary.types import (
    BudgetGuard,
    Candidate,
    Evaluator,
    EvolutionReport,
    ExperimentLedger,
    MutationRecord,
    Reviewer,
    Verdict,
    config_hash,
)

__all__ = [
    "EvolutionaryLoop",
    "SimpleBudgetGuard",
    # genome operators
    "random_config",
    "mutate",
    "crossover",
    "FUSION_CHOICES",
    "RETRIEVER_POOL",
    "RERANKER_TOP_K_RANGE",
    # selection
    "ScorePolicy",
    "composite_score",
    "qualifies",
    "select",
    "DEFAULT_PRIMARY_METRIC",
    "DEFAULT_DELTA_THRESHOLD",
    # types + protocols
    "Candidate",
    "MutationRecord",
    "EvolutionReport",
    "Verdict",
    "config_hash",
    "Evaluator",
    "Reviewer",
    "BudgetGuard",
    "ExperimentLedger",
]
