"""Compaction round-trip test (SPEC §6.5, epic ka-hvm acceptance).

Round-trips a 50-message state through compaction and verifies it preserves what
the agent needs (goal, plan, last turns, all citations), drops what it should
(raw tool outputs, uncited candidates), and that the compacted state can still
continue the conversation through a real orchestrator graph. Fully offline.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from common.schemas import (
    Chunk,
    Citation,
    Plan,
    PlanStep,
    Query,
    RetrievalCandidate,
    RetrievalResult,
    Source,
)
from harness.citation import CitationEnforcer, CitedDraft, CitedSegment
from harness.compaction import (
    CompactionConfig,
    HierarchicalSummarizationCompactor,
    OffloadToMemoryCompactor,
    SelectiveRetentionCompactor,
    estimate_state_tokens,
)
from harness.context import DefaultPacker
from harness.memory import (
    LayeredMemory,
    LongTermMemory,
    MemoryExtractor,
    SessionMemory,
    WorkingMemory,
)
from harness.memory.extraction import ExtractedFact, ExtractionResult
from harness.orchestrator import OrchestratorDeps, build_orchestrator, initial_state
from harness.planning import ReactPlanner
from knowledge_index.embedding import HashEmbedder
from knowledge_index.indexing import QdrantIndex


def _state_50():
    messages = []
    for i in range(25):
        messages.append(HumanMessage(content=f"question {i} about topic {i}"))
        messages.append(AIMessage(content=f"answer {i} with details about topic {i}"))
    cit = Citation(source=Source(doc_id="d1", chunk_id="cited-1"), claim_span=(0, 5))
    cited = RetrievalCandidate(
        chunk=Chunk(chunk_id="cited-1", doc_id="d1", text="kept"),
        score=0.9,
        retriever="dense",
        rank=1,
    )
    uncited = RetrievalCandidate(
        chunk=Chunk(chunk_id="uncited-9", doc_id="d2", text="drop"),
        score=0.4,
        retriever="dense",
        rank=2,
    )
    return {
        "question": "the original goal",
        "messages": messages,
        "plan": Plan(goal="the original goal", steps=[PlanStep(id="act", description="do it")]),
        "citations": [cit],
        "candidates": [cited, uncited],
        "retrieval_results": [
            RetrievalResult(candidates=[cited], query=Query(raw="q"), trace_id=uuid4())
        ],
        "scratchpad": "",
    }


async def test_selective_retention_roundtrip_preserves_and_drops():
    state = _state_50()
    assert len(state["messages"]) == 50
    comp = SelectiveRetentionCompactor(CompactionConfig(max_tokens=10, keep_last_turns=3))

    assert await comp.should_compact(state) is True
    new = await comp.compact(state)

    # preserved: goal, plan, all citations, last 3 turns (6 msgs) + 1 summary note
    assert new["question"] == "the original goal"
    assert new["plan"].goal == "the original goal"
    assert len(new["citations"]) == 1
    assert len(new["messages"]) == 7
    assert new["messages"][-1].content == "answer 24 with details about topic 24"
    assert "compaction summary" in str(new["messages"][0].content)

    # dropped: raw retrieval results, uncited candidates
    assert new["retrieval_results"] == []
    assert [c.chunk.chunk_id for c in new["candidates"]] == ["cited-1"]

    # the compacted state is materially smaller and no longer needs compaction
    assert estimate_state_tokens(new) < estimate_state_tokens(state)
    light = SelectiveRetentionCompactor(CompactionConfig(max_tokens=10_000, keep_last_turns=3))
    assert await light.should_compact(new) is False


async def test_hierarchical_summarization_uses_summarizer():
    async def summarizer(texts: list[str]) -> str:
        return f"summary of {len(texts)} messages"

    comp = HierarchicalSummarizationCompactor(
        CompactionConfig(max_tokens=10, keep_last_turns=3), summarizer_fn=summarizer
    )
    new = await comp.compact(_state_50())
    assert "summary of 44 messages" in str(new["messages"][0].content)


async def test_offload_to_memory_extracts_before_dropping():
    async def fake_extract(text: str) -> ExtractionResult:
        return ExtractionResult(facts=[ExtractedFact(key="topic0", value="about topic 0")])

    index = QdrantIndex("ka_memory_longterm", dim=48, location=":memory:")
    memory = LayeredMemory(
        working=WorkingMemory(),
        session=SessionMemory("s", path=":memory:"),
        long_term=LongTermMemory(index, HashEmbedder(dim=48)),
        extractor=MemoryExtractor(extract_fn=fake_extract),
    )
    comp = OffloadToMemoryCompactor(memory, CompactionConfig(max_tokens=10, keep_last_turns=3))
    new = await comp.compact(_state_50())
    assert "offloaded 1 durable fact" in str(new["messages"][0].content)
    # the fact is now durable in long-term memory
    hits = await memory.read("topic 0", "long_term")
    assert any(h.key == "topic0" for h in hits)


async def test_compacted_state_continues_through_orchestrator():
    # Build a real orchestrator and feed it the compacted state for a new turn.
    state = _state_50()
    compacted = await SelectiveRetentionCompactor(
        CompactionConfig(max_tokens=10, keep_last_turns=3)
    ).compact(state)

    class _Pipe:
        cost = 0.0

        async def retrieve(self, query: Query, k: int) -> RetrievalResult:
            cand = RetrievalCandidate(
                chunk=Chunk(chunk_id="new-1", doc_id="d", text="fresh fact"),
                score=1.0,
                retriever="dense",
                rank=1,
            )
            return RetrievalResult(candidates=[cand], query=query, trace_id=uuid4())

    async def draft_fn(question, candidates):
        top = candidates[0].chunk
        return CitedDraft(segments=[CitedSegment(text=top.text, citation_ids=[top.chunk_id])])

    deps = OrchestratorDeps(
        pipeline=_Pipe(),
        enforcer=CitationEnforcer(draft_fn=draft_fn),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
    )
    app = build_orchestrator(deps)

    # Continue: start a fresh turn but carry forward the compacted history.
    cont = initial_state("a follow-up question", budget_usd=1.0, max_hops=1, k=5)
    cont["messages"] = compacted["messages"]
    final = await app.ainvoke(cont, {"configurable": {"thread_id": str(uuid4())}})
    assert final["result"].text == "fresh fact"
    assert final["result"].citations  # still produces grounded output
