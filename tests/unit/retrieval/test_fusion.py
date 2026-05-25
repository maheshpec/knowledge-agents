"""Tests for RRF and weighted fusion (SPEC §7.6.4)."""

import pytest

from knowledge_index.retrieval.fusion import DEFAULT_RRF_K, RRFFuser, WeightedFuser


@pytest.mark.asyncio
async def test_rrf_rewards_agreement_across_lists(candidate_fn):
    # "b" is mid-rank in both lists; "a" is top of one only.
    list1 = [candidate_fn("a", 9.0, "dense", 1), candidate_fn("b", 5.0, "dense", 2)]
    list2 = [candidate_fn("c", 9.0, "sparse_bm25", 1), candidate_fn("b", 5.0, "sparse_bm25", 2)]

    fused = await RRFFuser().fuse([list1, list2])

    ids = [c.chunk.chunk_id for c in fused]
    # b appears in both -> 1/(60+2)+1/(60+2) beats a single 1/(60+1).
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}
    assert [c.rank for c in fused] == [1, 2, 3]


@pytest.mark.asyncio
async def test_rrf_uses_rank_not_score(candidate_fn):
    # A huge score at a poor rank must lose to a modest score at rank 1.
    list1 = [candidate_fn("top", 0.01, "dense", 1)]
    list2 = [
        candidate_fn("x", 1000.0, "sparse_bm25", 1),
        candidate_fn("y", 999.0, "sparse_bm25", 2),
        candidate_fn("top", 0.0, "sparse_bm25", 3),
    ]
    fused = await RRFFuser().fuse([list1, list2])
    # "top" is rank-1 in list1 + rank-3 in list2 -> highest combined reciprocal rank.
    assert fused[0].chunk.chunk_id == "top"


@pytest.mark.asyncio
async def test_rrf_k_must_be_positive():
    with pytest.raises(ValueError):
        RRFFuser(k=0)


@pytest.mark.asyncio
async def test_rrf_default_k_is_60():
    assert DEFAULT_RRF_K == 60


@pytest.mark.asyncio
async def test_weighted_fuser_respects_weights(candidate_fn):
    list_dense = [candidate_fn("d1", 10.0, "dense", 1), candidate_fn("d2", 0.0, "dense", 2)]
    list_sparse = [candidate_fn("s1", 10.0, "sparse_bm25", 1)]

    # Heavily weight sparse: its sole top-normalized (1.0) doc should win.
    fused = await WeightedFuser(weights={"dense": 0.1, "sparse_bm25": 5.0}).fuse(
        [list_dense, list_sparse]
    )
    assert fused[0].chunk.chunk_id == "s1"


@pytest.mark.asyncio
async def test_weighted_fuser_normalizes_flat_scores(candidate_fn):
    # All-equal scores normalize to 1.0 without dividing by zero.
    flat = [candidate_fn("a", 3.0, "dense", 1), candidate_fn("b", 3.0, "dense", 2)]
    fused = await WeightedFuser().fuse([flat])
    assert {c.chunk.chunk_id for c in fused} == {"a", "b"}
