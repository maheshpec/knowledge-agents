"""Tests for core schemas (SPEC §5)."""

from uuid import uuid4

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


def test_chunk_minimal():
    c = Chunk(chunk_id="x", doc_id="y", text="z")
    assert c.chunk_id == "x"
    assert c.acl == []
    assert c.context is None


def test_query_defaults():
    q = Query(raw="what is X?")
    assert q.intent == "unknown"
    assert q.rewrites == []


def test_retrieval_result_roundtrips_json():
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="hello")
    cand = RetrievalCandidate(chunk=chunk, score=0.9, retriever="dense", rank=1)
    result = RetrievalResult(candidates=[cand], query=Query(raw="q"), trace_id=uuid4())
    dumped = result.model_dump_json()
    reloaded = RetrievalResult.model_validate_json(dumped)
    assert reloaded.candidates[0].chunk.chunk_id == "c1"
    assert reloaded.cost == 0.0


def test_plan_forward_ref_resolves():
    plan = Plan(goal="g", steps=[PlanStep(id="s1", description="do thing")])
    assert plan.status == "draft"
    assert plan.steps[0].status == "pending"


def test_citation_requires_claim_span():
    src = Source(doc_id="d", chunk_id="c")
    cit = Citation(source=src, claim_span=(0, 10))
    assert cit.claim_span == (0, 10)
