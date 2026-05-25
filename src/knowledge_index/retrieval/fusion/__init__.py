"""Fusion: RRF (default) and score-normalized weighted sum (SPEC §7.6.4)."""

from knowledge_index.retrieval.fusion.base import Fuser, candidate_key
from knowledge_index.retrieval.fusion.rrf import DEFAULT_RRF_K, RRFFuser
from knowledge_index.retrieval.fusion.weighted import WeightedFuser

__all__ = [
    "Fuser",
    "candidate_key",
    "DEFAULT_RRF_K",
    "RRFFuser",
    "WeightedFuser",
]
