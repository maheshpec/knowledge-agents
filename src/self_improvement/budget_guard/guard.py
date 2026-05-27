"""Budget guard for the self-improvement loop (SPEC §8.5).

Enforces hard ceilings on an evolutionary run and trips a one-way kill switch the
instant any is breached:

- ``max_generations`` per run,
- ``max_compute_hours_per_gen`` per generation,
- ``max_usd_per_run`` cumulative spend per run,
- ``daily_usd_ceiling`` cumulative spend across all runs in a calendar day.

The guard is the single chokepoint the loop must call before doing expensive
work (``start_generation``) and after incurring cost/compute (``record_*``).
Once tripped it stays tripped: every guarded call raises :class:`BudgetExceeded`
so an in-flight run cannot sneak past a breached ceiling.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime

from pydantic import BaseModel


class BudgetConfig(BaseModel):
    """Hard ceilings for the loop (SPEC §8.5)."""

    max_generations: int = 20
    max_compute_hours_per_gen: float = 4.0
    max_usd_per_run: float = 50.0
    daily_usd_ceiling: float = 200.0


class BudgetExceeded(RuntimeError):
    """Raised when a ceiling is breached; carries the limit that tripped."""

    def __init__(self, ceiling: str, detail: str) -> None:
        super().__init__(f"budget ceiling breached [{ceiling}]: {detail}")
        self.ceiling = ceiling
        self.detail = detail


class BudgetGuard:
    """Stateful enforcer of :class:`BudgetConfig` ceilings with a kill switch.

    ``now`` is injectable so the daily ceiling can be tested deterministically.
    """

    def __init__(
        self,
        config: BudgetConfig | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config or BudgetConfig()
        self._now = now or (lambda: datetime.now(UTC))
        self.tripped: bool = False
        self.trip_reason: str | None = None
        # Per-run accounting.
        self._generations: dict[str, int] = defaultdict(int)
        self._run_spend: dict[str, float] = defaultdict(float)
        self._gen_compute_s: dict[tuple[str, int], float] = defaultdict(float)
        # Per-calendar-day accounting across all runs.
        self._daily_spend: dict[date, float] = defaultdict(float)

    # --- kill switch ----------------------------------------------------

    def _trip(self, ceiling: str, detail: str) -> BudgetExceeded:
        self.tripped = True
        self.trip_reason = f"{ceiling}: {detail}"
        return BudgetExceeded(ceiling, detail)

    def assert_live(self) -> None:
        """Raise if the kill switch has already tripped."""
        if self.tripped:
            raise BudgetExceeded("kill_switch", self.trip_reason or "tripped")

    # --- guarded operations --------------------------------------------

    def start_generation(self, run_id: str) -> int:
        """Account for a new generation; raise if it exceeds ``max_generations``.

        Returns the 1-based generation number just started.
        """
        self.assert_live()
        n = self._generations[run_id] + 1
        if n > self.config.max_generations:
            raise self._trip(
                "max_generations",
                f"run '{run_id}' would start generation {n} > {self.config.max_generations}",
            )
        self._generations[run_id] = n
        return n

    def record_spend(self, run_id: str, usd: float) -> None:
        """Add spend; raise if it breaches the per-run or daily ceiling."""
        self.assert_live()
        if usd < 0:
            raise ValueError("spend must be non-negative")
        self._run_spend[run_id] += usd
        today = self._now().date()
        self._daily_spend[today] += usd
        if self._run_spend[run_id] > self.config.max_usd_per_run:
            raise self._trip(
                "max_usd_per_run",
                f"run '{run_id}' spend ${self._run_spend[run_id]:.2f} > "
                f"${self.config.max_usd_per_run:.2f}",
            )
        if self._daily_spend[today] > self.config.daily_usd_ceiling:
            raise self._trip(
                "daily_usd_ceiling",
                f"{today} spend ${self._daily_spend[today]:.2f} > "
                f"${self.config.daily_usd_ceiling:.2f}",
            )

    def record_compute(self, run_id: str, generation: int, seconds: float) -> None:
        """Add compute for a generation; raise if it breaches the hourly ceiling."""
        self.assert_live()
        if seconds < 0:
            raise ValueError("compute seconds must be non-negative")
        key = (run_id, generation)
        self._gen_compute_s[key] += seconds
        limit_s = self.config.max_compute_hours_per_gen * 3600.0
        if self._gen_compute_s[key] > limit_s:
            hours = self._gen_compute_s[key] / 3600.0
            raise self._trip(
                "max_compute_hours_per_gen",
                f"run '{run_id}' gen {generation} compute {hours:.2f}h > "
                f"{self.config.max_compute_hours_per_gen:.2f}h",
            )

    # --- introspection --------------------------------------------------

    def run_spend(self, run_id: str) -> float:
        return self._run_spend[run_id]

    def daily_spend(self, day: date | None = None) -> float:
        return self._daily_spend[day or self._now().date()]

    def generations(self, run_id: str) -> int:
        return self._generations[run_id]


__all__ = ["BudgetConfig", "BudgetExceeded", "BudgetGuard"]
