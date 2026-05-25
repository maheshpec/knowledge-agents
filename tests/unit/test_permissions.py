"""Tests for permission gates and the LangGraph interrupt integration (SPEC §6.10)."""

from typing import Any, TypedDict

import pytest

from harness.permissions import (
    ApprovalRequest,
    BudgetGate,
    ConcurrencyGate,
    WriteGate,
    default_gates,
    evaluate_gates,
    make_gate_node,
)

# --- gate predicates ---


def test_write_gate_pauses_on_external_write():
    gate = WriteGate()
    assert gate.should_pause({"pending_action": {"type": "external_write"}})
    assert not gate.should_pause({"pending_action": {"type": "tool"}})
    assert not gate.should_pause({})  # no pending action


def test_budget_gate_pauses_above_threshold():
    gate = BudgetGate(threshold=0.25)
    assert gate.should_pause({"pending_action": {"type": "tool", "cost": 0.50}})
    assert not gate.should_pause({"pending_action": {"type": "tool", "cost": 0.10}})


def test_concurrency_gate_pauses_beyond_cap():
    gate = ConcurrencyGate(max_concurrent=3)
    # 2 active + 2 requested = 4 > 3 -> pause
    assert gate.should_pause(
        {"pending_action": {"type": "spawn", "spawn_count": 2}, "active_subagents": 2}
    )
    # 1 active + 1 requested = 2 <= 3 -> ok
    assert not gate.should_pause(
        {"pending_action": {"type": "spawn", "spawn_count": 1}, "active_subagents": 1}
    )
    # non-spawn action never trips this gate
    assert not gate.should_pause({"pending_action": {"type": "tool", "cost": 99.0}})


def test_build_request_carries_gate_and_payload():
    gate = WriteGate()
    state = {"pending_action": {"type": "external_write", "description": "send email"}}
    request = gate.build_request(state)
    assert isinstance(request, ApprovalRequest)
    assert request.gate == "write"
    assert request.risk == "high"
    assert request.payload["type"] == "external_write"
    assert "send email" in request.action


def test_evaluate_gates_returns_first_tripped():
    gates = default_gates(budget_threshold=0.25, max_concurrent=3)
    # external_write trips WriteGate (first in the list)
    req = evaluate_gates(gates, {"pending_action": {"type": "external_write"}})
    assert req is not None and req.gate == "write"
    # high-cost tool trips BudgetGate
    req2 = evaluate_gates(gates, {"pending_action": {"type": "tool", "cost": 1.0}})
    assert req2 is not None and req2.gate == "budget"
    # benign action trips nothing
    assert evaluate_gates(gates, {"pending_action": {"type": "tool", "cost": 0.01}}) is None


# --- LangGraph interrupt integration ---


class _S(TypedDict, total=False):
    pending_action: Any
    active_subagents: int
    approval: Any
    approval_denied: bool
    finished: bool


def _build_app(gates):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    async def finish(state: _S) -> dict:
        return {"finished": True}

    graph = StateGraph(_S)
    graph.add_node("gate", make_gate_node(gates))
    graph.add_node("finish", finish)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=MemorySaver())


@pytest.mark.asyncio
async def test_orchestrator_pauses_at_write_gate_then_resumes_on_approval():
    from langgraph.types import Command

    app = _build_app([WriteGate()])
    config = {"configurable": {"thread_id": "approve-1"}}

    paused = await app.ainvoke(
        {"pending_action": {"type": "external_write", "description": "send email"}}, config
    )
    assert "__interrupt__" in paused  # graph paused awaiting approval
    assert not paused.get("finished")

    final = await app.ainvoke(Command(resume={"request_id": "r", "approved": True}), config)
    assert final["finished"] is True
    assert final["approval"]["approved"] is True
    assert final.get("pending_action") is None


@pytest.mark.asyncio
async def test_gate_resumes_on_denial_and_flags_state():
    from langgraph.types import Command

    app = _build_app([WriteGate()])
    config = {"configurable": {"thread_id": "deny-1"}}

    await app.ainvoke({"pending_action": {"type": "external_write"}}, config)
    final = await app.ainvoke(Command(resume={"request_id": "r", "approved": False}), config)
    assert final["finished"] is True
    assert final["approval_denied"] is True


@pytest.mark.asyncio
async def test_gate_passes_through_when_no_action_pending():
    app = _build_app([WriteGate()])
    config = {"configurable": {"thread_id": "passthrough-1"}}
    final = await app.ainvoke({"pending_action": None}, config)
    assert final["finished"] is True
    assert "__interrupt__" not in final
