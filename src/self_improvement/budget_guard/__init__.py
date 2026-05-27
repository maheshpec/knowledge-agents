"""Budget guard (SPEC §8.5): hard ceilings + kill switch for the loop."""

from __future__ import annotations

from self_improvement.budget_guard.guard import (
    BudgetConfig,
    BudgetExceeded,
    BudgetGuard,
)

__all__ = ["BudgetConfig", "BudgetExceeded", "BudgetGuard"]
