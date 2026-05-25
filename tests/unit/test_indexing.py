"""Tests for the Qdrant hybrid index (SPEC §7.5).

Uses qdrant-client's in-process ``:memory:`` mode and the dependency-free
HashingBM25 sparse encoder, so these run with no server and no network.
"""

import pytest

from common.schemas import Chunk
from knowledge_index.embedding import HashEmbedder
from knowledge_index.enrichment.base import embedding_text
from knowledge_index.indexing import QdrantIndex


async def _make_index(chunks: list[Chunk], dim: int = 64) -> tuple[QdrantIndex, HashEmbedder]:
    emb = HashEmbedder(dim=dim)
    vecs = await emb.embed_documents([embedding_text(c) for c in chunks])
    for c, v in zip(chunks, vecs, strict=True):
        c.embedding = v
    index = QdrantIndex("test", dim=dim, location=":memory:")
    await index.upsert(chunks)
    return index, emb


async def test_dense_and_sparse_search_roundtrip():
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="the cat sat on the mat"),
        Chunk(chunk_id="c2", doc_id="d2", text="quantum chromodynamics and gluons"),
    ]
    index, emb = await _make_index(chunks)
    assert await index.count() == 2

    qvec = await emb.embed_query("a cat on a mat")
    dense = await index.search_dense(qvec, k=2, filters={})
    assert dense
    assert dense[0].chunk.chunk_id == "c1"
    assert dense[0].retriever == "dense"

    sparse = await index.search_sparse("cat mat", k=2, filters={})
    assert sparse
    assert sparse[0].chunk.chunk_id == "c1"
    assert sparse[0].retriever == "sparse_bm25"


async def test_acl_enforced_via_payload_filter():
    chunks = [
        Chunk(chunk_id="priv", doc_id="d1", text="secret budget numbers", acl=["alice"]),
        Chunk(chunk_id="pub", doc_id="d2", text="public budget overview", acl=[]),
    ]
    index, emb = await _make_index(chunks)
    qvec = await emb.embed_query("budget")

    alice = await index.search_dense(qvec, k=10, filters={"user_principals": ["alice"]})
    ids_alice = {c.chunk.chunk_id for c in alice}
    assert "priv" in ids_alice and "pub" in ids_alice

    bob = await index.search_dense(qvec, k=10, filters={"user_principals": ["bob"]})
    ids_bob = {c.chunk.chunk_id for c in bob}
    assert "priv" not in ids_bob  # bob cannot see alice's private chunk
    assert "pub" in ids_bob  # empty-acl chunk is public


async def test_doc_id_filter():
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="alpha content here"),
        Chunk(chunk_id="c2", doc_id="d2", text="alpha content here too"),
    ]
    index, emb = await _make_index(chunks)
    qvec = await emb.embed_query("alpha content")
    res = await index.search_dense(qvec, k=10, filters={"doc_id": "d1"})
    assert {c.chunk.doc_id for c in res} == {"d1"}


async def test_delete_removes_points():
    chunks = [Chunk(chunk_id="c1", doc_id="d1", text="to be deleted")]
    index, _ = await _make_index(chunks)
    assert await index.count() == 1
    await index.delete(["c1"])
    assert await index.count() == 0


async def test_snapshot_and_restore():
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="snapshot me one"),
        Chunk(chunk_id="c2", doc_id="d2", text="snapshot me two"),
    ]
    index, _ = await _make_index(chunks)
    ref = await index.snapshot()
    assert ref.count == 2
    await index.delete(["c1", "c2"])
    assert await index.count() == 0
    await index.restore(ref)
    assert await index.count() == 2


async def test_upsert_requires_embedding():
    index = QdrantIndex("noemb", dim=8, location=":memory:")
    with pytest.raises(Exception):
        await index.upsert([Chunk(chunk_id="x", doc_id="d", text="no vector")])
