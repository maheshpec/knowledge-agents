"""Phase 2 acceptance test (SPEC §10 Phase 2, epic ka-2ba).

Drives the *full* Phase 2 orchestrator offline — query router (G), Plan-and-Execute
planner (G), skills (F), long-term memory (E), permission gate (F), and
clean-context sub-agents (E) — against a controlled corpus, so it needs no API
keys or servers and runs in CI.

Asserts the Phase 2 acceptance contract from the epic:

- A complex comparison query is decomposed by the planner into 3 sub-tasks.
- The two independent legs are delegated to 2 sub-agents, each doing its own
  retrieval; with the parent's own retrieval that is 3 total retrieval calls.
- Citations are preserved across the sub-agent boundary: the evidence the
  delegates grounded on is re-cited in the parent's final answer.
- Sub-agents parallelize: the delegated run completes in roughly the slowest
  leg's time, not the sum (latency assertion skipped on CI).
- No regression: the Phase 1 ReAct path still produces a cited answer.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from uuid import uuid4

from common.schemas import (
    Chunk,
    Query,
    RetrievalCandidate,
    RetrievalResult,
)
from common.types import MemoryItem, Skill, SkillManifest
from harness import answer
from harness.citation import CitationEnforcer, CitedDraft, CitedSegment
from harness.context import DefaultPacker
from harness.orchestrator import OrchestratorDeps, build_orchestrator, initial_state
from harness.permissions.gates import default_gates
from harness.planning import ReactPlanner, TodoListPlanner
from harness.subagents.orchestrator_agent import orchestrator_agent_fn
from knowledge_index.retrieval.routers.base import RouteDecision

COMPLEX_QUERY = "Compare Python vs Rust across performance, safety, and ecosystem"

# Per-query distinctive evidence so we can prove a *sub-agent's* chunk reaches the
# parent's final answer. A sub-task mentioning "python" retrieves the python chunk;
# "rust" the rust chunk; the parent's full comparison query gets a general chunk.
_CHUNKS = {
    "python": Chunk(chunk_id="c_python", doc_id="d_py", text="Python is dynamically typed."),
    "rust": Chunk(chunk_id="c_rust", doc_id="d_rs", text="Rust guarantees memory safety."),
    "general": Chunk(chunk_id="c_general", doc_id="d_g", text="Both are popular languages."),
}

# Per-retrieval latency, so two parallel sub-agent legs visibly beat the serial sum.
_RETRIEVE_DELAY_S = 0.1


class FakePipeline:
    """Query-keyed retrieval that sleeps (to expose parallelism) and counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def _chunk_for(self, raw: str) -> Chunk:
        q = raw.lower()
        has_py, has_rs = "python" in q, "rust" in q
        if has_py and not has_rs:
            return _CHUNKS["python"]
        if has_rs and not has_py:
            return _CHUNKS["rust"]
        return _CHUNKS["general"]

    async def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.calls += 1
        await asyncio.sleep(_RETRIEVE_DELAY_S)
        cand = RetrievalCandidate(
            chunk=self._chunk_for(query.raw), score=1.0, retriever="dense", rank=0
        )
        return RetrievalResult(candidates=[cand], query=query, trace_id=uuid4(), cost=0.0)


class StubRouter:
    """Offline router: a 'compare' query is a high-complexity comparison; else lookup."""

    name = "stub_router"

    async def route(self, query: Query) -> RouteDecision:
        if "compare" in query.raw.lower():
            return RouteDecision(
                strategy="iterative", intent="comparison", expected_complexity="high"
            )
        return RouteDecision(strategy="hybrid", intent="lookup", expected_complexity="low")


async def _plan_completer(prompt: str) -> str:
    """Decompose a comparison goal into 3 steps (2 independent legs + synthesis)."""
    goal = prompt.split("Goal:")[-1].lower()
    if "compare" in goal and "python" in goal and "rust" in goal:
        return json.dumps(
            [
                {
                    "id": "py",
                    "description": "Research Python performance and safety",
                    "depends_on": [],
                },
                {
                    "id": "rs",
                    "description": "Research Rust performance and safety",
                    "depends_on": [],
                },
                {
                    "id": "syn",
                    "description": "Synthesize the comparison",
                    "depends_on": ["py", "rs"],
                },
            ]
        )
    # A leaf research task: a single step with no dependents -> no further delegation.
    return json.dumps([{"id": "act", "description": goal.strip()[:60], "depends_on": []}])


async def _cite_all(question: str, candidates: list[RetrievalCandidate]) -> CitedDraft:
    """Cite every candidate so sub-agent-sourced chunks appear in the final answer."""
    if not candidates:
        return CitedDraft(refused=True, refusal_reason="no evidence")
    segments = [
        CitedSegment(text=c.chunk.text, citation_ids=[c.chunk.chunk_id]) for c in candidates
    ]
    return CitedDraft(segments=segments)


def _phase2_deps(
    pipeline: FakePipeline, *, skills=None, manifests=None, memory=None
) -> OrchestratorDeps:
    deps = OrchestratorDeps(
        pipeline=pipeline,
        enforcer=CitationEnforcer(draft_fn=_cite_all),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
        planner_mode="todo_list",
        todo_planner=TodoListPlanner(_plan_completer),
        router=StubRouter(),
        skills=skills,
        skill_manifests=manifests or [],
        memory=memory,
        gates=default_gates(),  # ConcurrencyGate cap 3 — 2 spawns pass through
        subagent_budget_usd=0.25,
    )
    deps.agent_fn = orchestrator_agent_fn(deps, k=5, max_hops=1)
    return deps


async def _run(deps: OrchestratorDeps, question: str, **kw):
    app = build_orchestrator(deps)
    state = initial_state(question, budget_usd=2.0, k=5, max_hops=1, **kw)
    cfg = {"configurable": {"thread_id": str(uuid4())}}
    return await app.ainvoke(state, cfg)


async def test_complex_query_plans_delegates_and_preserves_citations():
    pipe = FakePipeline()
    deps = _phase2_deps(pipe)

    t0 = time.perf_counter()
    final = await _run(deps, COMPLEX_QUERY)
    elapsed = time.perf_counter() - t0

    # 1. Planner decomposed the goal into 3 sub-tasks.
    assert final["plan"] is not None
    assert len(final["plan"].steps) == 3, "comparison should decompose into 3 steps"

    # 2. Two independent legs were delegated to sub-agents, each succeeding.
    results = final["subagent_results"]
    assert len(results) == 2, "two independent legs -> two sub-agents"
    assert all(r.ok for r in results)
    assert final["delegated"] is True

    # 3. Three total retrieval calls: one per sub-agent + the parent's own.
    assert pipe.calls == 3, f"expected 3 retrieval calls, got {pipe.calls}"

    # 4. Citations preserved across the sub-agent boundary: the python and rust
    #    chunks the delegates grounded on are re-cited in the parent's answer.
    cited_ids = {c.source.chunk_id for c in final["result"].citations}
    assert "c_python" in cited_ids, "sub-agent (python) evidence lost"
    assert "c_rust" in cited_ids, "sub-agent (rust) evidence lost"
    assert final["result"].text

    # 5. Sub-agents parallelize: the two delegated legs overlap, so the run beats
    #    the serial sum of all three retrievals. (Skipped on CI — timing noisy.)
    if not os.getenv("CI"):
        assert elapsed < 3 * _RETRIEVE_DELAY_S, (
            f"delegated run {elapsed:.3f}s not faster than serial "
            f"{3 * _RETRIEVE_DELAY_S:.3f}s — sub-agents did not parallelize"
        )


class StubMemory:
    """Minimal Memory.read stand-in returning seeded long-term hits."""

    def __init__(self, items: list[MemoryItem]) -> None:
        self._items = items

    async def read(self, query: str, scope: str, k: int = 5) -> list[MemoryItem]:
        return self._items[:k] if scope == "long_term" else []


async def test_skills_selected_and_memory_read_at_context_step():
    from harness.skills.registry import SkillRegistry, keyword_overlap_classify

    pipe = FakePipeline()
    registry = SkillRegistry(classify=keyword_overlap_classify)
    # Pre-seed a loadable skill without touching disk by stubbing the loader.
    manifest = SkillManifest(
        name="compare",
        description="comparing technologies",
        when_to_use="compare X vs Y",
        path="/x",
    )
    skill = Skill(name="compare", instructions="Lay out a criteria table.", description="cmp")
    registry._manifests = {"compare": manifest}
    registry.load = lambda name: skill  # type: ignore[assignment]

    memory = StubMemory(
        [MemoryItem(key="pref", value="user prefers concise tables", scope="long_term")]
    )

    deps = _phase2_deps(pipe, skills=registry, manifests=[manifest], memory=memory)
    final = await _run(deps, COMPLEX_QUERY)

    # Context step selected the skill (F) and read long-term memory (E).
    assert [s.name for s in final["selected_skills"]] == ["compare"]
    assert any("user prefers" in str(item.value) for item in final["memory_hits"])
    assert final["result"].citations


async def test_permission_gate_pauses_spawn_when_concurrency_cap_tripped():
    # SPEC §6.10: a gate trips before the spawn and the graph pauses via interrupt.
    pipe = FakePipeline()
    deps = _phase2_deps(pipe)
    deps.gates = default_gates(max_concurrent=1)  # 2 spawns > cap -> pause

    final = await _run(deps, COMPLEX_QUERY)
    # LangGraph surfaces the pause as an __interrupt__ on the returned state; the
    # spawn has not happened yet, so no sub-agent results and no parent answer.
    assert "__interrupt__" in final
    assert not final.get("subagent_results")


async def test_phase1_react_path_still_answers():
    # No regression: with planner_mode='react' and no Phase 2 components wired,
    # the orchestrator runs the Phase 1 loop and still returns a cited answer.
    pipe = FakePipeline()
    deps = OrchestratorDeps(
        pipeline=pipe,
        enforcer=CitationEnforcer(draft_fn=_cite_all),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
    )
    result = await answer(
        "What does Python offer?", deps=deps, app=build_orchestrator(deps), budget_usd=1.0, k=5
    )
    assert result.text
    assert result.citations
    assert pipe.calls == 1  # single parent retrieval, no delegation
