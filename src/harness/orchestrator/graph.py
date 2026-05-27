"""LangGraph orchestrator (SPEC §6.1).

The full node set (Phase 1 + Phase 2I, ka-2ba):

    plan → context → route → {retrieve | sub-agent | answer}
                              retrieve  → observe → compact? → route
                              sub-agent → observe → compact? → route
                              answer    → finalize → END

``context`` runs the query router (G), selects skills (F), and reads long-term
memory (E) once per query. ``route`` is budget-aware: it finalizes early when
budget is spent, delegates to sub-agents when the router flags a delegation
intent, retrieves while hops remain, then answers. A ``sub-agent`` node spawns
clean-context delegates (E) in parallel and folds their citations back into the
parent evidence. The ``compact?`` edge consults the compactor (E) after each
observe. A ``gate`` node (F) can pause before a spawn via LangGraph ``interrupt``.

Every Phase 2 component is optional on :class:`OrchestratorDeps`; when none are
wired the graph collapses to the Phase 1 ``plan → route → {retrieve|answer}``
loop, so Phase 1 callers are unaffected.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph

from common.schemas import Chunk, GenerationResult, Query, RetrievalCandidate
from common.types import BudgetSpec
from harness.budget.tracker import BudgetTracker
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


def _candidate_from_citation(cit: Any, fallback_text: str = "") -> RetrievalCandidate:
    """Rebuild a citable candidate from a sub-agent's :class:`Citation`.

    Sub-agents return a :class:`GenerationResult`; its citations point at the
    chunks they grounded on. We lift those chunks back into the parent's
    candidate set so the parent can re-cite them — this is how citations are
    *preserved across the sub-agent boundary* (SPEC §6.4 acceptance). The lifted
    chunk keeps the original ``chunk_id`` (so the source pointer survives) and
    carries the citation's quote, or the sub-agent's answer text as a fallback,
    as its body (a candidate with empty text would be dropped at enforcement).
    """
    src = cit.source
    chunk = Chunk(
        chunk_id=src.chunk_id,
        doc_id=src.doc_id,
        parent_id=src.parent_id,
        text=cit.quote or fallback_text or src.chunk_id,
        metadata={"doc_title": src.title} if src.title else {},
    )
    return RetrievalCandidate(chunk=chunk, score=1.0, retriever="subagent", rank=0)


def delegatable_steps(plan: Any) -> list[Any]:
    """The plan steps a parent should delegate to parallel sub-agents.

    These are the *independent* steps (no ``depends_on``): in a comparison plan
    ("research X", "research Y", "synthesize") the two research legs have no
    dependencies and run as sub-agents, while the synthesis step (which depends
    on them) is performed by the parent's answer node.
    """
    if plan is None or not getattr(plan, "steps", None):
        return []
    independent = [s for s in plan.steps if not s.depends_on]
    # Only worth delegating when there is more than one independent leg to
    # parallelize *and* at least one dependent step left for the parent.
    dependent = [s for s in plan.steps if s.depends_on]
    if len(independent) >= 2 and dependent:
        return independent
    return []


def async_sqlite_saver(path: str = ":memory:") -> Any:
    """Build an ``AsyncSqliteSaver`` for checkpointing (SPEC §6.1).

    The orchestrator runs async (``ainvoke``), so the sync ``SqliteSaver`` cannot
    serve it — ``AsyncSqliteSaver`` (backed by ``aiosqlite``) is the persistent
    checkpointer. Pass the returned saver to :func:`build_orchestrator`.
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    return AsyncSqliteSaver(aiosqlite.connect(path))


def build_orchestrator(deps: OrchestratorDeps, *, checkpointer: Any = None) -> Any:
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
        # planner_mode (SPEC §6.2): 'react' acts step-at-a-time, 'todo_list'
        # decomposes the goal up front. deps.active_planner() resolves which.
        plan = await deps.active_planner().plan(state["question"], ctx)
        return {"plan": plan}

    @traced(span_name="orchestrator.context")
    async def context_node(state: OrchestratorState) -> dict:
        """Route the query, then select skills + read memory for the pack step.

        Runs once after plan. The router's intent drives both delegation (read in
        ``route_decision``) and skill selection; long-term memory hits are pulled
        for the answer preamble (SPEC §6.3, §6.6, §6.8, §7.6.1). Every branch is
        a no-op when its component is unwired, so Phase 1 deps skip straight on.
        """
        query = Query(raw=state["question"], user_principals=state.get("user_principals", []))
        delta: dict = {}
        if deps.router is not None:
            decision = await deps.router.route(query)
            delta["route_decision"] = decision
            query = query.model_copy(update={"intent": decision.intent})
        if deps.skills is not None and deps.skill_manifests:
            skills = await deps.skills.select(query, deps.skill_manifests, deps.skill_k)
            delta["selected_skills"] = skills
            _log.info("orchestrator.context", skills=[s.name for s in skills])
        if deps.memory is not None:
            hits = await deps.memory.read(state["question"], "long_term", deps.memory_k)
            delta["memory_hits"] = hits
        return delta

    async def route_node(state: OrchestratorState) -> dict:
        # Pure decision point; the conditional edge reads state. Kept as a node so
        # the sub-agent / permission branches can attach here.
        return {}

    def _should_delegate(state: OrchestratorState) -> bool:
        if not (deps.allow_delegation and deps.agent_fn is not None):
            return False
        if state.get("delegated") or state.get("delegation_depth", 0) >= deps.max_delegation_depth:
            return False
        decision = state.get("route_decision")
        intent = decision.intent if decision is not None else None
        if intent not in deps.delegation_intents:
            return False
        return bool(delegatable_steps(state.get("plan")))

    def route_decision(state: OrchestratorState) -> str:
        if state["budget_remaining"] <= MIN_ANSWER_BUDGET:
            _log.info("orchestrator.route", decision="answer", reason="budget_exhausted")
            return "answer"
        if _should_delegate(state):
            _log.info("orchestrator.route", decision="sub-agent")
            return "sub-agent"
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
            plan = await deps.active_planner().adapt(plan, {"candidates": n, "failed": n == 0})
        return {"scratchpad": scratchpad, "plan": plan}

    @traced(span_name="orchestrator.sub_agent")
    async def subagent_node(state: OrchestratorState) -> dict:
        """Delegate the plan's independent legs to parallel sub-agents (SPEC §6.4).

        Each leg runs as a clean-context sub-agent (its own LangGraph instance,
        no parent history) carved a child budget from the parent. Sub-agents run
        concurrently via ``spawn_all`` so wall-time tracks the slowest leg, not
        the sum. Their citations are lifted back into the parent candidate set so
        the parent can re-cite the evidence its delegates found.
        """
        from harness.subagents.base import SubAgentTask
        from harness.subagents.runner import spawn_all

        steps = delegatable_steps(state.get("plan"))
        parent = BudgetTracker(state["budget_remaining"])
        tasks = [
            SubAgentTask(
                task=s.description,
                return_schema=GenerationResult,
                budget=BudgetSpec(max_usd=min(deps.subagent_budget_usd, state["budget_remaining"])),
            )
            for s in steps
        ]
        results = await spawn_all(tasks, parent, deps.agent_fn)  # type: ignore[arg-type]

        candidates = state.get("candidates", [])
        for res in results:
            if res.ok and res.output:
                gen = GenerationResult.model_validate(res.output)
                lifted = [_candidate_from_citation(c, gen.text) for c in gen.citations]
                if lifted:
                    candidates = _accumulate(candidates, lifted)
        spent = parent.consumed
        _log.info(
            "orchestrator.sub_agent",
            spawned=len(results),
            ok=sum(1 for r in results if r.ok),
            cost=spent,
        )
        return {
            "candidates": candidates,
            "subagent_results": results,
            "delegated": True,
            "pending_action": None,
            "budget_remaining": max(0.0, state["budget_remaining"] - spent),
        }

    @traced(span_name="orchestrator.gate")
    async def gate_node(state: OrchestratorState) -> dict:
        """Permission gate before a spawn (SPEC §6.10).

        Surfaces the intended spawn as a ``pending_action`` and evaluates the
        configured gates. If one trips, the graph pauses via LangGraph
        ``interrupt`` (checkpointed) until a human resumes with an
        :class:`ApprovalResponse`. On denial the spawn is skipped and the plan is
        marked delegated so the loop routes around it.
        """
        from harness.permissions.base import evaluate_gates

        spawn_count = len(delegatable_steps(state.get("plan")))
        action = {
            "type": "spawn",
            "description": f"spawn {spawn_count} research sub-agent(s)",
            "spawn_count": spawn_count,
        }
        scoped = {**state, "pending_action": action}
        request = evaluate_gates(list(deps.gates or []), scoped)
        if request is None:
            return {"pending_action": action}

        from langgraph.types import interrupt

        _log.info("orchestrator.gate.pause", gate=request.gate, reason=request.reason)
        resume = interrupt(request.model_dump())
        approved = resume.get("approved", False) if isinstance(resume, dict) else bool(resume)
        if approved:
            return {"pending_action": action, "approval": {"gate": request.gate, "approved": True}}
        _log.info("orchestrator.gate.denied", gate=request.gate)
        return {"pending_action": None, "approval_denied": True, "delegated": True}

    @traced(span_name="orchestrator.compact")
    async def compact_node(state: OrchestratorState) -> dict:
        """Shrink history when it nears the model limit (SPEC §6.5)."""
        compacted = await deps.compactor.compact(state)  # type: ignore[union-attr]
        _log.info("orchestrator.compact", messages=len(compacted.get("messages", []) or []))
        return dict(compacted)

    async def compact_decision(state: OrchestratorState) -> str:
        if deps.compactor is not None and await deps.compactor.should_compact(state):
            return "compact"
        return "route"

    def gate_to_spawn(state: OrchestratorState) -> str:
        # After the gate: spawn unless approval was denied.
        return "route" if state.get("approval_denied") else "sub-agent"

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
        # Phase 2 context pack: prepend selected skills + memory hits as guidance
        # so the drafter sees them alongside the evidence (SPEC §6.6).
        preamble = deps.packer.render_preamble(
            state.get("selected_skills", []), state.get("memory_hits", [])
        )
        question = f"{preamble}\n\n{state['question']}" if preamble else state["question"]
        draft = await deps.enforcer.draft(question, candidates, trace_id=trace_id)
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
    graph.add_node("context", context_node)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("observe", observe_node)
    graph.add_node("sub_agent", subagent_node)
    graph.add_node("answer", answer_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "context")
    graph.add_edge("context", "route")

    # ``route`` fans out to retrieve / sub-agent / answer. The "sub-agent" branch
    # passes through the permission gate first when gates are configured.
    subagent_target = "gate" if deps.gates else "sub_agent"
    graph.add_conditional_edges(
        "route",
        route_decision,
        {"retrieve": "retrieve", "sub-agent": subagent_target, "answer": "answer"},
    )
    graph.add_edge("retrieve", "observe")
    graph.add_edge("sub_agent", "observe")

    if deps.gates:
        graph.add_node("gate", gate_node)
        graph.add_conditional_edges(
            "gate", gate_to_spawn, {"sub-agent": "sub_agent", "route": "route"}
        )

    # observe → compact? → route. The compact node only exists when a compactor
    # is wired; otherwise observe routes straight back.
    if deps.compactor is not None:
        graph.add_node("compact", compact_node)
        graph.add_conditional_edges(
            "observe", compact_decision, {"compact": "compact", "route": "route"}
        )
        graph.add_edge("compact", "route")
    else:
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
    delegation_depth: int = 0,
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
        # Phase 2 fields (ka-2ba): default to the inert Phase 1 values.
        route_decision=None,
        selected_skills=[],
        memory_hits=[],
        subagent_results=[],
        delegated=False,
        delegation_depth=delegation_depth,
        pending_action=None,
        approval=None,
        approval_denied=False,
        active_subagents=0,
    )


__all__ = ["build_orchestrator", "async_sqlite_saver", "initial_state", "MIN_ANSWER_BUDGET"]
