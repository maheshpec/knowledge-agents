"""Tests for EntityGraphBuilder over chunks (SPEC §7.7)."""

from __future__ import annotations

from common.schemas import Chunk
from knowledge_index.graph.builder import COOCCURS, EntityGraphBuilder


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=f"doc-{cid}", text=text)


async def test_build_links_entities_to_chunks():
    chunks = [_chunk("c1", "Acme Corp acquired Globex.")]
    store = await EntityGraphBuilder().build(chunks)
    assert await store.has_entity("acme corp")
    assert await store.has_entity("globex")
    linked = {c.chunk_id for c in await store.chunks_for("acme corp")}
    assert "c1" in linked


async def test_build_adds_typed_and_cooccurrence_edges():
    chunks = [_chunk("c1", "Acme Corp acquired Globex.")]
    store = await EntityGraphBuilder().build(chunks)
    rels = await store.neighbors("acme corp")
    preds = {r.predicate for r in rels}
    assert "acquired" in preds
    assert COOCCURS in preds  # both entities co-occur in c1


async def test_build_traversal_connects_entities_across_chunks():
    # c1 mentions A and B; c2 mentions B and C. Traversing from A reaches C in 2 hops
    # (A—B co-occurrence in c1, B—C co-occurrence in c2) — the multi-hop property.
    chunks = [
        _chunk("c1", "Acme Corp acquired Globex."),
        _chunk("c2", "Globex was founded by Jane Doe."),
    ]
    store = await EntityGraphBuilder().build(chunks)
    reached = await store.traverse(["acme corp"], depth=2)
    assert "jane doe" in reached


async def test_cooccurrence_can_be_disabled():
    chunks = [_chunk("c1", "Acme Corp acquired Globex.")]
    store = await EntityGraphBuilder(cooccurrence=False).build(chunks)
    preds = {r.predicate for r in await store.neighbors("acme corp")}
    assert COOCCURS not in preds
    assert "acquired" in preds


async def test_build_empty_chunks():
    store = await EntityGraphBuilder().build([])
    assert store.entity_count() == 0
