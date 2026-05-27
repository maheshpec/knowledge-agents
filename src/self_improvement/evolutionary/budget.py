"""A simple in-process budget guard (SPEC §8.5).

Enforces hard ceilings on generations, compute-seconds, and USD spend for one
evolutionary run. Implements the :class:`BudgetGuard` protocol so the Phase 4O
budget guard (with daily cross-run ceilings + a real kill switch) can replace it
without the loop changing.
"""

from __future__ import annotations


class SimpleBudgetGuard:
    """Trip when any of the per-run ceilings is breached (SPEC §8.5)."""

    def __init__(
        self,
        *,
        max_generations: int | None = None,
        max_cost_usd: float | None = None,
        max_compute_seconds: float | None = None,
    ) -> None:
        self.max_generations = max_generations
        self.max_cost_usd = max_cost_usd
        self.max_compute_seconds = max_compute_seconds
        self.generations = 0
        self.cost_usd = 0.0
        self.compute_seconds = 0.0

    def charge(self, *, cost_usd: float = 0.0, compute_seconds: float = 0.0) -> None:
        if cost_usd < 0 or compute_seconds < 0:
            raise ValueError("charges must be non-negative")
        self.cost_usd += cost_usd
        self.compute_seconds += compute_seconds

    def tick_generation(self) -> None:
        self.generations += 1

    def exhausted(self) -> bool:
        if self.max_generations is not None and self.generations >= self.max_generations:
            return True
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            return True
        if (
            self.max_compute_seconds is not None
            and self.compute_seconds >= self.max_compute_seconds
        ):
            return True
        return False


__all__ = ["SimpleBudgetGuard"]
