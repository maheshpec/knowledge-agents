"""Tests for the budget tracker (SPEC §6.11)."""

import pytest

from common.errors import BudgetExceeded
from harness.budget import BudgetTracker


def test_reserve_and_consume():
    t = BudgetTracker(10.0)
    grant = t.reserve(3.0)
    assert t.available() == 7.0
    t.consume(grant, 2.0)
    assert t.remaining() == 8.0
    assert t.available() == 8.0  # reservation released after settle


def test_reserve_beyond_limit_raises():
    t = BudgetTracker(5.0)
    with pytest.raises(BudgetExceeded):
        t.reserve(6.0)


def test_consume_cannot_double_settle():
    t = BudgetTracker(10.0)
    g = t.reserve(1.0)
    t.consume(g, 1.0)
    with pytest.raises(ValueError):
        t.consume(g, 1.0)


def test_child_budget_bubbles_spend_to_parent():
    parent = BudgetTracker(10.0)
    child = parent.child_budget(0.5)  # 5.0 fenced off
    assert child.limit == 5.0
    cg = child.reserve(2.0)
    child.consume(cg, 2.0)
    # child spend booked against both child and parent
    assert child.remaining() == 3.0
    assert parent.consumed == 2.0


def test_child_budget_cannot_overspend_parent():
    parent = BudgetTracker(4.0)
    parent.child_budget(1.0)  # fences the entire remaining budget
    with pytest.raises(BudgetExceeded):
        parent.reserve(1.0)  # nothing left for the parent


def test_remaining_never_negative():
    t = BudgetTracker(1.0)
    g = t.reserve(1.0)
    t.consume(g, 1.0)
    assert t.remaining() == 0.0
