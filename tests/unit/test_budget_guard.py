"""Tests for the budget guard (SPEC §8.5)."""

from datetime import UTC, datetime, timedelta

import pytest

from self_improvement.budget_guard import BudgetConfig, BudgetExceeded, BudgetGuard


def test_generations_within_limit_ok():
    guard = BudgetGuard(BudgetConfig(max_generations=3))
    assert [guard.start_generation("r") for _ in range(3)] == [1, 2, 3]
    assert guard.generations("r") == 3


def test_max_generations_trips_kill_switch():
    guard = BudgetGuard(BudgetConfig(max_generations=2))
    guard.start_generation("r")
    guard.start_generation("r")
    with pytest.raises(BudgetExceeded) as ei:
        guard.start_generation("r")
    assert ei.value.ceiling == "max_generations"
    assert guard.tripped is True


def test_max_usd_per_run_trips():
    guard = BudgetGuard(BudgetConfig(max_usd_per_run=10.0))
    guard.record_spend("r", 6.0)
    with pytest.raises(BudgetExceeded) as ei:
        guard.record_spend("r", 5.0)  # cumulative 11 > 10
    assert ei.value.ceiling == "max_usd_per_run"
    assert guard.run_spend("r") == pytest.approx(11.0)


def test_per_run_spend_is_isolated_across_runs():
    guard = BudgetGuard(BudgetConfig(max_usd_per_run=10.0, daily_usd_ceiling=1000.0))
    guard.record_spend("r1", 9.0)
    guard.record_spend("r2", 9.0)  # different run, own ceiling
    assert not guard.tripped


def test_daily_ceiling_trips_across_runs():
    guard = BudgetGuard(BudgetConfig(max_usd_per_run=1000.0, daily_usd_ceiling=15.0))
    guard.record_spend("r1", 10.0)
    with pytest.raises(BudgetExceeded) as ei:
        guard.record_spend("r2", 6.0)  # daily 16 > 15
    assert ei.value.ceiling == "daily_usd_ceiling"


def test_daily_ceiling_resets_next_day():
    clock = {"t": datetime(2026, 5, 27, tzinfo=UTC)}
    guard = BudgetGuard(
        BudgetConfig(max_usd_per_run=1000.0, daily_usd_ceiling=15.0),
        now=lambda: clock["t"],
    )
    guard.record_spend("r", 14.0)
    clock["t"] = clock["t"] + timedelta(days=1)  # new calendar day
    guard.record_spend("r", 14.0)  # fresh daily bucket
    assert not guard.tripped
    assert guard.daily_spend(datetime(2026, 5, 28, tzinfo=UTC).date()) == pytest.approx(14.0)


def test_max_compute_hours_per_gen_trips():
    guard = BudgetGuard(BudgetConfig(max_compute_hours_per_gen=1.0))
    guard.record_compute("r", 1, 1800)  # 0.5h
    with pytest.raises(BudgetExceeded) as ei:
        guard.record_compute("r", 1, 2400)  # cumulative 1.17h > 1h
    assert ei.value.ceiling == "max_compute_hours_per_gen"


def test_compute_ceiling_is_per_generation():
    guard = BudgetGuard(BudgetConfig(max_compute_hours_per_gen=1.0))
    guard.record_compute("r", 1, 3000)  # gen 1
    guard.record_compute("r", 2, 3000)  # gen 2 — separate bucket
    assert not guard.tripped


def test_kill_switch_is_one_way():
    guard = BudgetGuard(BudgetConfig(max_usd_per_run=1.0))
    with pytest.raises(BudgetExceeded):
        guard.record_spend("r", 2.0)
    # Every subsequent guarded call raises, even ones that would otherwise pass.
    with pytest.raises(BudgetExceeded):
        guard.start_generation("r")
    with pytest.raises(BudgetExceeded):
        guard.record_compute("r", 1, 1.0)
    with pytest.raises(BudgetExceeded):
        guard.assert_live()


def test_negative_amounts_rejected():
    guard = BudgetGuard()
    with pytest.raises(ValueError):
        guard.record_spend("r", -1.0)
    with pytest.raises(ValueError):
        guard.record_compute("r", 1, -1.0)
