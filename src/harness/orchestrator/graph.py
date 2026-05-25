"""LangGraph orchestrator (SPEC §6.1).

The Phase 1 node set:

    plan → route → {retrieve → observe → route | answer} → finalize

``route`` is budget-aware: it finalizes early when budget is spent, retrieves
while hops remain, then answers. ``answer`` drafts a cited answer over the packed
evidence; ``finalize`` enforces citations into the :class:`GenerationResult`.
Sub-agent, tool, compaction and permission nodes are deferred to Phase 2; their
seams are noted inline.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph

from common.schemas import GenerationResult, Query, RetrievalCandidate
from harness.citation.base import CitedDraft, Strictness
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from harness.orchestrator.state import OrchestratorDeps, OrchestratorState
from harness.planning.base import PlanningContext

_log = get_logger("harness.orchestrator")

# Minimum budget (USD) considered enough to attempt one answer LLM call.
MIN_ANSWER_BUDGET = 1e-6


def _accumulate(
    existing: list[RetrievalCandidate], new: list[RetrievalCandidate]
) -> list[RetrievalCandidate]:
    """Merge candidate lists, deduping by chunk_id and keeping the best score."""
    best: dict[str, RetrievalCandidate] = {c.chunk.chunk_id: c for c in existing}
    for cand in new:
        prev = best.get(cand.chunk.chunk_id)
        if prev is None or cand.score > prev.score:
            best[cand.chunk.chunk_id] = cand
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def async_sqlite_saver(path: str = ":memory:"):  # type: ignore[no-untyped-def]
    """Build an ``AsyncSqliteSaver`` for checkpointing (SPEC §6.1).

    The orchestrator runs async (``ainvoke``), so the sync ``SqliteSaver`` cannot
    serve it — ``AsyncSqliteSaver`` (backed by ``aiosqlite``) is the persistent
    checkpointer. Pass the returned saver to :func:`build_orchestrator`.
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    return AsyncSqliteSaver(aiosqlite.connect(path))


def build_orchestrator(deps: OrchestratorDeps, *, checkpointer: object | None = None):  # type: ignore[no-untyped-def]
    """Build and compile the Phase 1 orchestrator graph.

    ``checkpointer`` defaults to an in-process ``MemorySaver`` (async-safe, no
    extra deps). For cross-process persistence pass :func:`async_sqlite_saver`
    (SPEC §6.1 — the async-correct SqliteSaver variant).
    """

    @traced(span_name="orchestrator.plan")
    async def plan_node(state: OrchestratorState) -> dict:
        ctx = PlanningContext(
            budget_remaining=state["budget_remaining"],
            max_hops=state["max_hops"],
            user_principals=state.get("user_principals", []),
        )
        plan = await deps.planner.plan(state["question"], ctx)
        return {"plan": plan}

    async def route_node(state: OrchestratorState) -> dict:
        # Pure decision point; the conditional edge reads state. Kept as a node so
        # Phase 2 can attach tool/sub-agent/permission branches here.
        return {}

    def route_decision(state: OrchestratorState) -> str:
        if state["budget_remaining"] <= MIN_ANSWER_BUDGET:
            _log.info("orchestrator.route", decision="answer", reason="budget_exhausted")
            return "answer"
        if state["hops"] < state["max_hops"]:
            _log.info("orchestrator.route", decision="retrieve", hop=state["hops"])
            return "retrieve"
        _log.info("orchestrator.route", decision="answer", hop=state["hops"])
        return "answer"

    @traced(span_name="orchestrator.retrieve")
    async def retrieve_node(state: OrchestratorState) -> dict:
        query = Query(raw=state["question"], user_principals=state.get("user_principals", []))
        result = await deps.pipeline.retrieve(query, state["k"])
        results = [*state.get("retrieval_results", []), result]
        candidates = _accumulate(state.get("candidates", []), result.candidates)
        remaining = max(0.0, state["budget_remaining"] - result.cost)
        return {
            "retrieval_results": results,
            "candidates": candidates,
            "budget_remaining": remaining,
            "hops": state["hops"] + 1,
        }

    @traced(span_name="orchestrator.observe")
    async def observe_node(state: OrchestratorState) -> dict:
        n = len(state.get("candidates", []))
        note = f"hop {state['hops']}: {n} candidates accumulated"
        scratchpad = "\n".join(filter(None, [state.get("scratchpad", ""), note]))
        plan = state.get("plan")
        if plan is not None:
            plan = await deps.planner.adapt(plan, {"candidates": n, "failed": n == 0})
        # Phase 2 seam: compaction check (if token estimate > threshold) goes here.
        return {"scratchpad": scratchpad, "plan": plan}

    @traced(span_name="orchestrator.answer")
    async def answer_node(state: OrchestratorState) -> dict:
        candidates = deps.packer.order_evidence(
            state.get("candidates", []), deps.context_budget_tokens
        )
        trace_id = UUID(state["trace_id"])
        if state["budget_remaining"] <= MIN_ANSWER_BUDGET:
            draft = CitedDraft(
                refused=True,
                refusal_reason="I ran out of budget before I could answer.",
            )
            return {"draft": draft, "candidates": candidates, "budget_exhausted": True}
        draft = await deps.enforcer.draft(state["question"], candidates, trace_id=trace_id)
        remaining = max(0.0, state["budget_remaining"] - deps.enforcer._last_cost)
        return {"draft": draft, "candidates": candidates, "budget_remaining": remaining}

    @traced(span_name="orchestrator.finalize")
    async def finalize_node(state: OrchestratorState) -> dict:
        trace_id = UUID(state["trace_id"])
        draft = state.get("draft")
        candidates = state.get("candidates", [])
        strictness: Strictness = state.get("strictness", "strict")
        if draft is None:
            result = GenerationResult(
                text="", citations=[], trace_id=trace_id, cost=0.0, tokens_in=0, tokens_out=0
            )
        else:
            result = deps.enforcer.enforce(draft, candidates, strictness, trace_id=trace_id)
        _log.info(
            "orchestrator.finalize",
            trace_id=state["trace_id"],
            citations=len(result.citations),
            chars=len(result.text),
        )
        return {"result": result, "citations": result.citations}

    graph = StateGraph(OrchestratorState)
    graph.add_node("plan", plan_node)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("observe", observe_node)
    graph.add_node("answer", answer_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "route")
    graph.add_conditional_edges(
        "route", route_decision, {"retrieve": "retrieve", "answer": "answer"}
    )
    graph.add_edge("retrieve", "observe")
    graph.add_edge("observe", "route")
    graph.add_edge("answer", "finalize")
    graph.add_edge("finalize", END)

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def initial_state(
    question: str,
    *,
    budget_usd: float,
    k: int = 10,
    max_hops: int = 1,
    strictness: Strictness = "strict",
    user_principals: list[str] | None = None,
    trace_id: UUID | None = None,
) -> OrchestratorState:
    """Build the starting :class:`OrchestratorState` for one question."""
    return OrchestratorState(
        question=question,
        plan=None,
        retrieval_results=[],
        candidates=[],
        draft=None,
        result=None,
        citations=[],
        scratchpad="",
        budget_limit=budget_usd,
        budget_remaining=budget_usd,
        budget_exhausted=False,
        hops=0,
        max_hops=max_hops,
        k=k,
        strictness=strictness,
        user_principals=user_principals or [],
        trace_id=str(trace_id or uuid4()),
    )


__all__ = ["build_orchestrator", "async_sqlite_saver", "initial_state", "MIN_ANSWER_BUDGET"]
