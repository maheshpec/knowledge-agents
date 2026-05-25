"""Tests for the three-tier cache (SPEC §6.12)."""

import time

from harness.cache import (
    EmbeddingCache,
    RetrievalCache,
    build_cached_system,
    cacheable_text_block,
    count_breakpoints,
    retrieval_cache_key,
)

# --- prompt cache (tier 1) ---


def test_cacheable_block_marks_control():
    block = cacheable_text_block("hi", cache=True)
    assert block["cache_control"] == {"type": "ephemeral"}
    plain = cacheable_text_block("hi", cache=False)
    assert "cache_control" not in plain


def test_build_cached_system_sets_one_breakpoint():
    blocks = build_cached_system(["frozen system", "skills"], cache_last=True)
    assert count_breakpoints(blocks) == 1
    assert "cache_control" in blocks[-1]


def test_build_cached_system_empty():
    assert build_cached_system([]) == []


# --- embedding cache (tier 2) ---


def test_embedding_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    assert cache.get("voyage-3-large", "hello") is None
    cache.put("voyage-3-large", "hello", [0.1, 0.2, 0.3])
    assert cache.get("voyage-3-large", "hello") == [0.1, 0.2, 0.3]
    # keyed on (model, text): different model is a miss
    assert cache.get("other-model", "hello") is None
    cache.close()


def test_embedding_cache_get_many(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    cache.put("m", "a", [1.0])
    hits = cache.get_many("m", ["a", "b"])
    assert hits == {"a": [1.0]}
    cache.close()


# --- retrieval cache (tier 3) ---


def test_retrieval_cache_key_stable_under_filter_order():
    k1 = retrieval_cache_key("q", "v1", {"a": 1, "b": 2})
    k2 = retrieval_cache_key("q", "v1", {"b": 2, "a": 1})
    assert k1 == k2


def test_retrieval_cache_hit_and_ttl_expiry():
    cache = RetrievalCache(max_size=8, ttl_seconds=0.05)
    cache.put("k", ["result"])
    assert cache.get("k") == ["result"]
    time.sleep(0.06)
    assert cache.get("k") is None  # expired


def test_retrieval_cache_lru_eviction():
    cache = RetrievalCache(max_size=2, ttl_seconds=100)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # touch a so b is LRU
    cache.put("c", 3)  # evicts b
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
