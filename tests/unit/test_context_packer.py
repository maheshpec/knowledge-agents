"""Tests for the context packer (SPEC §6.6)."""

from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from common.schemas import Chunk, Query, RetrievalCandidate, RetrievalResult
from harness.context import DefaultPacker, Skill, reorder_for_lost_in_middle
from harness.context.base import estimate_tokens


def _cand(cid: str, score: float, text: str = "x") -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=Chunk(chunk_id=cid, doc_id="d", text=text), score=score, retriever="dense", rank=0
    )


def _result(cands: list[RetrievalCandidate]) -> RetrievalResult:
    return RetrievalResult(candidates=cands, query=Query(raw="q"), trace_id=uuid4())


def test_reorder_puts_best_at_both_ends():
    ranked = [_cand(str(i), 10 - i) for i in range(5)]  # best→worst: 0,1,2,3,4
    ordered = reorder_for_lost_in_middle(ranked)
    ids = [c.chunk.chunk_id for c in ordered]
    assert ids[0] == "0"  # best at top
    assert ids[-1] == "1"  # second-best at bottom
    assert ids == ["0", "2", "4", "3", "1"]


def test_pack_orders_system_then_evidence_then_turns():
    packer = DefaultPacker()
    retrieval = _result([_cand("c1", 0.9, "alpha"), _cand("c2", 0.5, "beta")])
    msgs = packer.pack(
        system="SYS",
        skills=[],
        memory_hits=[],
        retrieval=retrieval,
        scratchpad="",
        messages=[HumanMessage(content="the question")],
        budget_tokens=10_000,
    )
    assert isinstance(msgs[0], SystemMessage)
    # system content is a cacheable block list
    assert isinstance(msgs[0].content, list)
    assert msgs[0].content[0]["cache_control"] == {"type": "ephemeral"}
    # evidence block references chunk ids for citing
    evidence = next(m for m in msgs if isinstance(m, SystemMessage) and "[c1]" in str(m.content))
    assert "[c2]" in str(evidence.content)
    # the conversation turn is last
    assert isinstance(msgs[-1], HumanMessage)


def test_pack_includes_skills_in_system():
    packer = DefaultPacker()
    msgs = packer.pack(
        system="SYS",
        skills=[Skill(name="search", instructions="do search")],
        memory_hits=[],
        retrieval=None,
        scratchpad="",
        messages=[],
        budget_tokens=1000,
    )
    assert "Skill: search" in msgs[0].content[0]["text"]


def test_budget_trims_evidence_worst_first():
    packer = DefaultPacker()
    cands = [_cand(f"c{i}", 1.0 - i * 0.1, "word " * 50) for i in range(5)]
    # tiny budget keeps only the strongest couple of candidates
    fit = packer.fit_candidates(cands, budget_tokens=estimate_tokens("word " * 50) + 1)
    assert len(fit) == 1
    assert fit[0].chunk.chunk_id == "c0"
