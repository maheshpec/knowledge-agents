"""Tests for post-processors: MMR, parent, dedup, reorder, spans (SPEC §7.6.6)."""

import pytest

from common.schemas import Chunk, Query, RetrievalCandidate
from knowledge_index.retrieval.post import (
    DeduplicatorPostProcessor,
    LostInTheMiddleReorder,
    MMRDiversifier,
    ParentExpander,
    SpanExtractor,
)
from knowledge_index.retrieval.post.base import cosine


def _cand(chunk: Chunk, score: float, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(chunk=chunk, score=score, retriever="rerank", rank=rank)


def _avg_pairwise_cosine(cands):
    vecs = [c.chunk.embedding for c in cands]
    pairs = [(i, j) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    if not pairs:
        return 0.0
    return sum(cosine(vecs[i], vecs[j]) for i, j in pairs) / len(pairs)


@pytest.mark.asyncio
async def test_mmr_beats_relevance_only_baseline_on_diversity(fake_embedder_cls, make_chunk_fn):
    # A and B are duplicates (most relevant); C is somewhat relevant but diverse.
    # Relevance-only top-2 = {A,B} (redundant); MMR should swap B for C.
    a = make_chunk_fn("A", "doc a", embedding=[0.9, 0.436, 0.0])
    b = make_chunk_fn("B", "doc b", embedding=[0.9, 0.436, 0.0])
    c = make_chunk_fn("C", "doc c", embedding=[0.7, 0.0, 0.714])
    cands = [_cand(a, 1.0, 1), _cand(b, 0.99, 2), _cand(c, 0.1, 3)]

    embedder = fake_embedder_cls({"find a": [1.0, 0.0, 0.0]})
    mmr = MMRDiversifier(embedder.embed_query, lambda_=0.5, top_k=2)
    selected = await mmr.process(Query(raw="find a"), cands)

    baseline = cands[:2]  # relevance-only top-2
    assert {c.chunk.chunk_id for c in selected} == {"A", "C"}
    assert _avg_pairwise_cosine(selected) <= _avg_pairwise_cosine(baseline)
    assert [c.rank for c in selected] == [1, 2]


@pytest.mark.asyncio
async def test_mmr_lambda_one_is_pure_relevance(fake_embedder_cls, make_chunk_fn):
    a = make_chunk_fn("A", "a", embedding=[1.0, 0.0, 0.0])
    b = make_chunk_fn("B", "b", embedding=[0.99, 0.01, 0.0])
    c = make_chunk_fn("C", "c", embedding=[0.0, 1.0, 0.0])
    cands = [_cand(a, 1.0, 1), _cand(b, 0.99, 2), _cand(c, 0.1, 3)]
    embedder = fake_embedder_cls({"q": [1.0, 0.0, 0.0]})
    mmr = MMRDiversifier(embedder.embed_query, lambda_=1.0, top_k=2)
    selected = await mmr.process(Query(raw="q"), cands)
    # Pure relevance keeps the two most query-similar (A, B).
    assert {c.chunk.chunk_id for c in selected} == {"A", "B"}


@pytest.mark.asyncio
async def test_mmr_rejects_bad_lambda(fake_embedder_cls):
    with pytest.raises(ValueError):
        MMRDiversifier(fake_embedder_cls().embed_query, lambda_=1.5)


@pytest.mark.asyncio
async def test_parent_expander_swaps_and_dedupes(make_chunk_fn):
    parent = make_chunk_fn("P", "the full parent passage")
    child1 = make_chunk_fn("c1", "child one", parent_id="P")
    child2 = make_chunk_fn("c2", "child two", parent_id="P")
    lonely = make_chunk_fn("c3", "no parent")

    parents = {"P": parent}

    async def fetch(pid):
        return parents.get(pid)

    cands = [_cand(child1, 0.9, 1), _cand(child2, 0.8, 2), _cand(lonely, 0.7, 3)]
    out = await ParentExpander(fetch).process(Query(raw="q"), cands)

    ids = [c.chunk.chunk_id for c in out]
    assert ids == ["P", "c3"]  # two children collapse to one parent
    assert out[0].score == 0.9  # higher child's score carried onto parent


@pytest.mark.asyncio
async def test_parent_expander_leaves_orphans(make_chunk_fn):
    async def fetch(pid):
        return None  # parent not found

    cands = [_cand(make_chunk_fn("c1", "x", parent_id="missing"), 1.0, 1)]
    out = await ParentExpander(fetch).process(Query(raw="q"), cands)
    assert out[0].chunk.chunk_id == "c1"


@pytest.mark.asyncio
async def test_deduplicator_by_id_and_text(make_chunk_fn):
    dup_text_a = make_chunk_fn("a", "Same Body Text")
    dup_text_b = make_chunk_fn("b", "same   body text")  # same after normalize
    same_id = make_chunk_fn("a", "different")  # duplicate id
    unique = make_chunk_fn("u", "unique content")

    cands = [
        _cand(dup_text_a, 0.9, 1),
        _cand(dup_text_b, 0.8, 2),
        _cand(same_id, 0.7, 3),
        _cand(unique, 0.6, 4),
    ]
    out = await DeduplicatorPostProcessor().process(Query(raw="q"), cands)
    ids = [c.chunk.chunk_id for c in out]
    assert ids == ["a", "u"]
    assert [c.rank for c in out] == [1, 2]


# --- lost-in-the-middle reorder ---


@pytest.mark.asyncio
async def test_lost_in_the_middle_puts_best_at_both_ends(make_chunk_fn):
    # Ranked best->worst: 1,2,3,4,5. Expect best (1) at top, 2nd (2) at bottom,
    # weakest (5) buried in the middle.
    cands = [_cand(make_chunk_fn(str(i), f"t{i}"), 1.0 - i * 0.1, i) for i in range(1, 6)]
    out = await LostInTheMiddleReorder().process(Query(raw="q"), cands)
    ids = [c.chunk.chunk_id for c in out]
    assert ids == ["1", "3", "5", "4", "2"]
    assert out[0].chunk.chunk_id == "1"  # most relevant at top
    assert out[-1].chunk.chunk_id == "2"  # next most relevant at bottom
    assert [c.rank for c in out] == [1, 2, 3, 4, 5]  # ranks renumbered


@pytest.mark.asyncio
async def test_lost_in_the_middle_noop_for_small_sets(make_chunk_fn):
    cands = [_cand(make_chunk_fn("a", "x"), 1.0, 1), _cand(make_chunk_fn("b", "y"), 0.9, 2)]
    out = await LostInTheMiddleReorder().process(Query(raw="q"), cands)
    assert [c.chunk.chunk_id for c in out] == ["a", "b"]  # unchanged


# --- span extraction ---


@pytest.mark.asyncio
async def test_span_extractor_trims_to_relevant_window(make_chunk_fn):
    text = (
        "Intro about unrelated administrivia. "
        "The mitochondria is the powerhouse of the cell. "
        "Mitochondria generate ATP through respiration. "
        "Closing remark about something else entirely."
    )
    chunk = make_chunk_fn("c1", text)
    out = await SpanExtractor(max_sentences=2).process(
        Query(raw="what do mitochondria generate?"), [_cand(chunk, 1.0, 1)]
    )
    extracted = out[0].chunk.text
    assert "mitochondria" in extracted.lower()
    assert "ATP" in extracted
    assert "administrivia" not in extracted  # irrelevant intro dropped
    # Original text and offsets preserved for traceability.
    assert out[0].chunk.metadata["span_original_text"] == text
    start, end = out[0].chunk.metadata["span_offsets"]
    assert text[start:end].strip() == extracted


@pytest.mark.asyncio
async def test_span_extractor_keeps_chunk_when_no_overlap(make_chunk_fn):
    text = "Alpha beta gamma. Delta epsilon zeta."
    chunk = make_chunk_fn("c1", text)
    out = await SpanExtractor().process(
        Query(raw="completely unrelated terms"), [_cand(chunk, 1.0, 1)]
    )
    assert out[0].chunk.text == text  # untouched
    assert "span_offsets" not in out[0].chunk.metadata


@pytest.mark.asyncio
async def test_span_extractor_rejects_bad_window():
    with pytest.raises(ValueError):
        SpanExtractor(max_sentences=0)
