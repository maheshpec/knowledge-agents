"""Tests for the citation enforcer (SPEC §6.13)."""

from common.schemas import Chunk, RetrievalCandidate
from harness.citation import CitationEnforcer, CitedDraft, CitedSegment


def _cand(cid: str, text: str = "body", title: str | None = None) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=Chunk(chunk_id=cid, doc_id="d", text=text, metadata={"doc_title": title}),
        score=1.0,
        retriever="dense",
        rank=1,
    )


def _draft() -> CitedDraft:
    return CitedDraft(
        segments=[
            CitedSegment(text="Cats purr.", citation_ids=["c1"]),
            CitedSegment(text="Dogs bark.", citation_ids=["ghost"]),  # hallucinated id
        ]
    )


def test_strict_drops_unsupported_and_rejects_hallucinated():
    enf = CitationEnforcer()
    cands = [_cand("c1", "cats"), _cand("c2", "dogs")]
    result = enf.enforce(_draft(), cands, "strict")
    assert result.text == "Cats purr."  # the hallucinated-only segment is dropped
    assert len(result.citations) == 1
    assert result.citations[0].source.chunk_id == "c1"
    # claim_span points at the supported span within the final prose
    start, end = result.citations[0].claim_span
    assert result.text[start:end] == "Cats purr."


def test_loose_tags_uncited():
    enf = CitationEnforcer()
    cands = [_cand("c1")]
    result = enf.enforce(_draft(), cands, "loose")
    assert "Cats purr." in result.text
    assert "[uncited]" in result.text  # the unsupported segment is kept but tagged


def test_off_keeps_all_segments():
    enf = CitationEnforcer()
    cands = [_cand("c1")]
    result = enf.enforce(_draft(), cands, "off")
    assert "Cats purr." in result.text
    assert "Dogs bark." in result.text
    # only the valid citation is recorded even in off mode
    assert {c.source.chunk_id for c in result.citations} == {"c1"}


def test_refused_draft_returns_reason():
    enf = CitationEnforcer()
    draft = CitedDraft(refused=True, refusal_reason="no supporting evidence")
    result = enf.enforce(draft, [_cand("c1")], "strict")
    assert result.text == "no supporting evidence"
    assert result.citations == []


def test_string_draft_is_treated_as_single_uncited_segment():
    enf = CitationEnforcer()
    result = enf.enforce("just prose", [_cand("c1")], "off")
    assert result.text == "just prose"
    assert result.citations == []


async def test_generate_uses_injected_draft_fn():
    async def fake(question, candidates):
        return CitedDraft(
            segments=[CitedSegment(text="answer", citation_ids=[candidates[0].chunk.chunk_id])]
        )

    enf = CitationEnforcer(draft_fn=fake)
    cands = [_cand("c1")]
    result = await enf.generate("q?", cands, strictness="strict")
    assert result.text == "answer"
    assert result.citations[0].source.chunk_id == "c1"


def test_every_citation_id_is_in_candidate_set():
    # Citation precision guard: no citation may reference a non-candidate.
    enf = CitationEnforcer()
    cands = [_cand("c1"), _cand("c2")]
    draft = CitedDraft(
        segments=[CitedSegment(text="claim", citation_ids=["c1", "c2", "hallucinated"])]
    )
    result = enf.enforce(draft, cands, "strict")
    cited = {c.source.chunk_id for c in result.citations}
    assert cited == {"c1", "c2"}
