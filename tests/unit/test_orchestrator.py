"""Tests for the orchestrator graph (SPEC §6.1)."""

from uuid import uuid4

from common.schemas import Chunk, Query, RetrievalCandidate, RetrievalResult
from harness.citation import CitationEnforcer, CitedDraft, CitedSegment
from harness.context import DefaultPacker
from harness.orchestrator import OrchestratorDeps, build_orchestrator, initial_state
from harness.orchestrator.graph import async_sqlite_saver
from harness.planning import ReactPlanner


class FakePipeline:
    """Returns preset candidates and counts how many times it was called."""

    def __init__(self, candidates: list[RetrievalCandidate], cost: float = 0.0) -> None:
        self._candidates = candidates
        self.cost = cost
        self.calls = 0

    async def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(
            candidates=self._candidates[:k], query=query, trace_id=uuid4(), cost=self.cost
        )


def _cands(n: int) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            chunk=Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=f"fact number {i}"),
            score=1.0 - i * 0.1,
            retriever="dense",
            rank=i,
        )
        for i in range(n)
    ]


async def _draft_top(question, candidates):
    if not candidates:
        return CitedDraft(refused=True, refusal_reason="no evidence found")
    top = candidates[0].chunk
    return CitedDraft(segments=[CitedSegment(text=top.text, citation_ids=[top.chunk_id])])


def _deps(pipeline) -> OrchestratorDeps:
    return OrchestratorDeps(
        pipeline=pipeline,
        enforcer=CitationEnforcer(draft_fn=_draft_top),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
    )


async def _run(deps, **state_kwargs):
    app = build_orchestrator(deps)
    state = initial_state("what is fact 0?", **state_kwargs)
    cfg = {"configurable": {"thread_id": str(uuid4())}}
    return await app.ainvoke(state, cfg)


async def test_single_retrieval_produces_cited_answer():
    pipe = FakePipeline(_cands(3))
    final = await _run(_deps(pipe), budget_usd=1.0, max_hops=1, k=5)
    assert pipe.calls == 1
    result = final["result"]
    assert result.text == "fact number 0"
    assert len(result.citations) == 1
    # citation precision: every cited id was actually retrieved
    retrieved_ids = {c.chunk.chunk_id for c in final["candidates"]}
    assert all(cit.source.chunk_id in retrieved_ids for cit in result.citations)


async def test_zero_retrieval_when_max_hops_zero():
    pipe = FakePipeline(_cands(3))
    final = await _run(_deps(pipe), budget_usd=1.0, max_hops=0, k=5)
    assert pipe.calls == 0
    assert final["result"].text == "no evidence found"


async def test_two_retrieval_hops():
    pipe = FakePipeline(_cands(3))
    final = await _run(_deps(pipe), budget_usd=1.0, max_hops=2, k=5)
    assert pipe.calls == 2
    assert final["hops"] == 2


async def test_budget_exhausted_finalizes_early():
    pipe = FakePipeline(_cands(3))
    final = await _run(_deps(pipe), budget_usd=0.0, max_hops=1, k=5)
    assert pipe.calls == 0  # never retrieves; routes straight to answer
    assert "budget" in final["result"].text.lower()
    assert final["result"].citations == []


async def test_runs_with_async_sqlite_checkpointer():
    # SPEC §6.1: the graph must run under the (async) SqliteSaver checkpointer.
    pipe = FakePipeline(_cands(3))
    app = build_orchestrator(_deps(pipe), checkpointer=async_sqlite_saver(":memory:"))
    thread_id = str(uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    final = await app.ainvoke(initial_state("what is fact 0?", budget_usd=1.0), cfg)
    assert final["result"].text == "fact number 0"
    # state was checkpointed under the thread id and is retrievable
    snapshot = await app.aget_state(cfg)
    assert snapshot.values["result"].text == "fact number 0"


# ---- Phase 5B: DCI routing + chained modes (SPEC §15.2/§15.3, ka-p1b) -------


class _FakeDCIExecutor:
    """Records calls and returns preset candidates as a RetrievalResult."""

    name = "fake_dci"

    def __init__(self, candidates: list[RetrievalCandidate], cost: float = 0.0) -> None:
        self._candidates = candidates
        self.cost = cost
        self.calls: list[Query] = []

    async def run(self, query: Query, k: int) -> RetrievalResult:
        self.calls.append(query)
        return RetrievalResult(
            candidates=self._candidates[:k], query=query, trace_id=uuid4(), cost=self.cost
        )


class _StubRouter:
    name = "stub"

    def __init__(self, strategy: str) -> None:
        from knowledge_index.retrieval.routers import RouteDecision

        self._decision = RouteDecision(strategy=strategy, intent="lookup")

    async def route(self, query):  # noqa: D401 - protocol method
        return self._decision


def _dci_cands(n: int) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            chunk=Chunk(chunk_id=f"g{i}", doc_id=f"d{i}", text=f"grep hit {i}"),
            score=1.0 - i * 0.1,
            retriever="dci",
            rank=i,
        )
        for i in range(n)
    ]


def _deps_with_dci(
    pipeline,
    dci_executor,
    *,
    strategy: str,
) -> OrchestratorDeps:
    return OrchestratorDeps(
        pipeline=pipeline,
        enforcer=CitationEnforcer(draft_fn=_draft_top),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
        router=_StubRouter(strategy),
        dci_executor=dci_executor,
        allow_delegation=False,
    )


async def test_dci_strategy_runs_dci_node_and_skips_retrieve():
    pipe = FakePipeline(_cands(3))
    dci = _FakeDCIExecutor(_dci_cands(3))
    deps = _deps_with_dci(pipe, dci, strategy="dci")
    final = await _run(deps, budget_usd=1.0, max_hops=1, k=5)
    assert len(dci.calls) == 1
    assert pipe.calls == 0
    # Candidates came from the DCI executor; the cited chunk_id is a DCI hit.
    assert final["result"].citations[0].source.chunk_id.startswith("g")


async def test_dci_then_vector_runs_dci_first_then_retrieve():
    pipe = FakePipeline(_cands(3))
    dci = _FakeDCIExecutor(_dci_cands(3))
    deps = _deps_with_dci(pipe, dci, strategy="dci_then_vector")
    final = await _run(deps, budget_usd=1.0, max_hops=2, k=5)
    assert len(dci.calls) == 1
    assert pipe.calls == 1
    assert final["hops"] == 2
    # Both retrievers contributed candidates to the final pool.
    retrievers = {c.retriever for c in final["candidates"]}
    assert retrievers == {"dci", "dense"}


async def test_vector_then_dci_runs_retrieve_first_then_dci():
    pipe = FakePipeline(_cands(3))
    dci = _FakeDCIExecutor(_dci_cands(3))
    deps = _deps_with_dci(pipe, dci, strategy="vector_then_dci")
    final = await _run(deps, budget_usd=1.0, max_hops=2, k=5)
    assert pipe.calls == 1
    assert len(dci.calls) == 1
    assert final["hops"] == 2
    retrievers = {c.retriever for c in final["candidates"]}
    assert retrievers == {"dci", "dense"}


async def test_dci_strategy_falls_back_when_no_executor_wired():
    # SPEC §15.3 graceful degradation: a DCI strategy must not crash when the
    # executor is missing — the orchestrator falls back to vector retrieve so
    # the answer still grounds on the vector hybrid pipeline.
    pipe = FakePipeline(_cands(3))
    deps = OrchestratorDeps(
        pipeline=pipe,
        enforcer=CitationEnforcer(draft_fn=_draft_top),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
        router=_StubRouter("dci"),
        dci_executor=None,
    )
    final = await _run(deps, budget_usd=1.0, max_hops=1, k=5)
    assert pipe.calls == 1
    assert final["result"].text == "fact number 0"


async def test_dci_tool_node_deducts_cost_from_budget():
    pipe = FakePipeline(_cands(3))
    dci = _FakeDCIExecutor(_dci_cands(3), cost=0.05)
    deps = _deps_with_dci(pipe, dci, strategy="dci")
    final = await _run(deps, budget_usd=1.0, max_hops=1, k=5)
    assert final["budget_remaining"] <= 1.0 - 0.05


async def test_planner_mode_todo_list_uses_todo_planner():
    # SPEC §6.2 / Phase 2G: planner_mode='todo_list' branches the plan node to
    # the injected todo_planner; 'react' (default) keeps the ReAct planner.
    import json

    from harness.planning import TodoListPlanner

    async def _plan_completer(prompt: str) -> str:
        return json.dumps([{"id": "s1", "description": "find fact 0", "depends_on": []}])

    pipe = FakePipeline(_cands(3))
    deps = OrchestratorDeps(
        pipeline=pipe,
        enforcer=CitationEnforcer(draft_fn=_draft_top),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
        planner_mode="todo_list",
        todo_planner=TodoListPlanner(_plan_completer),
    )
    assert deps.active_planner().name == "todo_list"
    final = await _run(deps, budget_usd=1.0, max_hops=1, k=5)
    plan = final["plan"]
    assert plan.steps[0].id == "s1"  # plan came from the todo planner
    assert final["result"].text == "fact number 0"
