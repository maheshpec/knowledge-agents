"""Scoring, qualification, and selection for the evolutionary loop (SPEC §8.2).

``composite_score`` reduces a candidate's metric dict to one number the loop
maximizes: a primary retrieval metric (default ``ndcg@10``) minus small cost and
latency penalties. ``select`` keeps the top-k by composite score, dropping any
candidate the reviewer rejected.

Qualification (``qualifies``) is the stricter, PR-gating bar from SPEC §8.2's
"critical constraints": a candidate must beat the seed baseline by at least the
delta threshold, must not regress on the rotating eval set (the Goodhart guard),
and must carry an ``accept`` verdict from the adversarial reviewer (§8.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from self_improvement.evolutionary.types import Candidate

DEFAULT_PRIMARY_METRIC = "ndcg@10"
DEFAULT_DELTA_THRESHOLD = 0.02  # e.g. +2% nDCG@10 to qualify (SPEC §8.2)


@dataclass(frozen=True)
class ScorePolicy:
    """How to collapse a metric dict into one maximizable composite score."""

    primary_metric: str = DEFAULT_PRIMARY_METRIC
    cost_weight: float = 0.0  # USD penalty per query
    latency_weight: float = 0.0  # penalty per second

    def score(self, candidate: Candidate) -> float:
        primary = candidate.metrics.get(self.primary_metric, 0.0)
        penalty = self.cost_weight * candidate.cost_usd + self.latency_weight * (
            candidate.compute_seconds
        )
        return primary - penalty


def composite_score(candidate: Candidate, policy: ScorePolicy | None = None) -> float:
    """Composite, maximizable score for one candidate (SPEC §8.2 step 6)."""
    return (policy or ScorePolicy()).score(candidate)


def qualifies(
    candidate: Candidate,
    *,
    baseline_score: float,
    policy: ScorePolicy | None = None,
    delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
) -> bool:
    """True if the candidate is a PR-worthy improvement (SPEC §8.2 constraints).

    Requires: composite score beats the baseline by ``delta_threshold``; reviewer
    verdict is ``accept``; and — when a rotating-set score is present — no
    regression there versus the primary metric (Goodhart guard).
    """
    policy = policy or ScorePolicy()
    if candidate.verdict != "accept":
        return False
    if composite_score(candidate, policy) - baseline_score < delta_threshold:
        return False
    if candidate.rotating_metrics is not None:
        rotating = candidate.rotating_metrics.get(policy.primary_metric, 0.0)
        primary = candidate.metrics.get(policy.primary_metric, 0.0)
        # The gain must not be an artifact of the dev set alone: the rotating set
        # must also clear the baseline (allowing a small noise band).
        if rotating - baseline_score < delta_threshold - 1e-9 and rotating < primary:
            return False
    return True


def select(
    candidates: list[Candidate], k: int, policy: ScorePolicy | None = None
) -> list[Candidate]:
    """Top-``k`` survivors by composite score, dropping reviewer-rejected ones."""
    policy = policy or ScorePolicy()
    survivors = [c for c in candidates if c.verdict != "reject"]
    for c in survivors:
        if c.score is None:
            c.score = composite_score(c, policy)
    survivors.sort(key=lambda c: c.score if c.score is not None else float("-inf"), reverse=True)
    return survivors[: max(0, k)]


__all__ = [
    "DEFAULT_PRIMARY_METRIC",
    "DEFAULT_DELTA_THRESHOLD",
    "ScorePolicy",
    "composite_score",
    "qualifies",
    "select",
]
