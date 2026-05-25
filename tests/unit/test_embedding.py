"""Tests for embedders + cache integration (SPEC §7.4, §6.12)."""

from harness.cache.embedding_cache import EmbeddingCache
from knowledge_index.embedding import HashEmbedder, build_embedder


async def test_hash_embedder_dim_and_determinism():
    emb = HashEmbedder(dim=64)
    assert emb.dim == 64
    a = await emb.embed_query("hello world")
    b = await emb.embed_query("hello world")
    assert a == b
    assert len(a) == 64
    # roughly unit-normalized
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


async def test_hash_embedder_documents_order_preserved():
    emb = HashEmbedder(dim=32)
    out = await emb.embed_documents(["a", "b", "c"])
    assert len(out) == 3
    assert out[0] != out[1]


async def test_embedder_uses_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    emb = HashEmbedder(dim=16, cache=cache)
    await emb.embed_documents(["repeated", "repeated", "unique"])
    # cached under the embedder's name
    assert cache.get(emb.name, "repeated") is not None
    assert cache.get(emb.name, "unique") is not None
    cache.close()


def test_build_embedder_registry():
    emb = build_embedder("hash", dim=128)
    assert emb.dim == 128
    voyage = build_embedder("voyage-3-large")
    assert voyage.name == "voyage-3-large"
    assert voyage.dim == 1024
