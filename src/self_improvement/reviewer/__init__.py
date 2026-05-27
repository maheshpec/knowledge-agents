"""Adversarial reviewer (SPEC §8.3): the gate that tries to invalidate a candidate.

Deterministic checks (leakage, narrow slice, noise band, perf regression) plus an
adversarial LLM pass produce a ``ReviewerVerdict`` whose verdict gates PR creation.
"""

from self_improvement.reviewer.models import (
    CheckResult,
    LineageEntry,
    LineageProvider,
    ReviewerVerdict,
    ReviewThresholds,
)
from self_improvement.reviewer.reviewer import (
    DEFAULT_REVIEWER_MODEL,
    REVIEW_PROMPT,
    AdversarialReviewer,
    CompletionFn,
    default_completion_fn,
    derive_baseline_and_seeds,
)

__all__ = [
    "AdversarialReviewer",
    "CompletionFn",
    "DEFAULT_REVIEWER_MODEL",
    "REVIEW_PROMPT",
    "default_completion_fn",
    "derive_baseline_and_seeds",
    "CheckResult",
    "ReviewerVerdict",
    "ReviewThresholds",
    "LineageEntry",
    "LineageProvider",
]
