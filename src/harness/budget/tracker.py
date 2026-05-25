"""Per-request token/cost budget tracking (SPEC §6.11).

The orchestrator checks `remaining()` at each route step; if low it finalizes
early with a "ran out of budget" caveat. Sub-agents receive a `child_budget`
carved from the parent, and their spend bubbles back up to the parent.
"""

from __future__ import annotations

import threading

from common.errors import BudgetExceeded
from common.types import BudgetGrant


class BudgetTracker:
    """Tracks reserved and consumed spend against a hard ceiling.

    Reservations let callers atomically claim headroom before an expensive call,
    then settle the actual cost afterward. Thread-safe so parallel sub-agents
    (SPEC §6.4) can share a parent tracker.
    """

    def __init__(self, limit_usd: float, *, parent: BudgetTracker | None = None) -> None:
        if limit_usd < 0:
            raise ValueError("budget limit must be non-negative")
        self._limit = limit_usd
        self._reserved = 0.0
        self._consumed = 0.0
        self._parent = parent
        self._lock = threading.RLock()

    # --- core API (SPEC §6.11 interface) ---

    def reserve(self, amount: float) -> BudgetGrant:
        """Reserve `amount` USD of headroom; raises BudgetExceeded if unavailable."""
        if amount < 0:
            raise ValueError("reserve amount must be non-negative")
        with self._lock:
            if amount > self._available_locked():
                raise BudgetExceeded(requested=amount, remaining=self._available_locked())
            self._reserved += amount
            return BudgetGrant(amount=amount)

    def consume(self, grant: BudgetGrant, actual: float) -> None:
        """Settle a reservation with the actual cost.

        Releases the reserved headroom and books the real spend. If `actual`
        exceeds what is left even after releasing the reservation, raises.
        """
        if actual < 0:
            raise ValueError("actual cost must be non-negative")
        with self._lock:
            if grant.settled:
                raise ValueError(f"grant {grant.grant_id} already settled")
            # Release the reservation first, then book the real spend.
            self._reserved -= grant.amount
            if actual > self._available_locked():
                # Re-reserve to keep accounting consistent before raising.
                self._reserved += grant.amount
                raise BudgetExceeded(requested=actual, remaining=self._available_locked())
            self._consumed += actual
            grant.settled = True
        # Bubble real spend up to the parent (SPEC §6.4 budget bubbling).
        if self._parent is not None:
            self._parent._absorb_child_spend(actual)

    def remaining(self) -> float:
        """USD left after consumed spend (ignores outstanding reservations)."""
        with self._lock:
            return max(0.0, self._limit - self._consumed)

    def available(self) -> float:
        """USD that can still be reserved (limit minus consumed AND reserved)."""
        with self._lock:
            return self._available_locked()

    def child_budget(self, fraction: float) -> BudgetTracker:
        """Carve a sub-budget for a sub-agent as a fraction of remaining budget."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        with self._lock:
            child_limit = self._available_locked() * fraction
            # Reserve the child's ceiling against this tracker so the parent
            # cannot double-spend it; the child settles against itself and
            # bubbles actual spend back up via _absorb_child_spend.
            self._reserved += child_limit
        return BudgetTracker(child_limit, parent=self)

    @property
    def consumed(self) -> float:
        with self._lock:
            return self._consumed

    @property
    def limit(self) -> float:
        return self._limit

    # --- internals ---

    def _available_locked(self) -> float:
        return max(0.0, self._limit - self._consumed - self._reserved)

    def _absorb_child_spend(self, actual: float) -> None:
        """Record a child's real spend against this (parent) tracker."""
        with self._lock:
            self._consumed += actual
        if self._parent is not None:
            self._parent._absorb_child_spend(actual)


__all__ = ["BudgetTracker"]
