"""Tests for graph stores: linkage + bounded traversal (SPEC §7.7)."""

from __future__ import annotations

from common.schemas import Chunk
from knowledge_index.graph.base import Entity, Relation
from knowledge_index.graph.store import InMemoryGraphStore, NetworkxGraphStore


def _chunk(cid: str, text: str = "t", acl: list[str] | None = None) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=f"doc-{cid}", text=text, acl=acl or [])


async def _line_graph() -> InMemoryGraphStore:
    """a — b — c — d (a chain of co-occurrence-style edges)."""
    store = InMemoryGraphStore()
    for a, b in [("a", "b"), ("b", "c"), ("c", "d")]:
        await store.add_relation(Relation(subject=a, predicate="rel", object=b))
    return store


async def test_add_relation_creates_endpoints():
    store = InMemoryGraphStore()
    await store.add_relation(Relation(subject="x", predicate="rel", object="y"))
    assert await store.has_entity("x")
    assert await store.has_entity("y")
    assert store.relation_count() == 1


async def test_traverse_respects_depth():
    store = await _line_graph()
    assert await store.traverse(["a"], depth=0) == {"a"}
    assert await store.traverse(["a"], depth=1) == {"a", "b"}
    assert await store.traverse(["a"], depth=2) == {"a", "b", "c"}
    assert await store.traverse(["a"], depth=3) == {"a", "b", "c", "d"}


async def test_traverse_unknown_seed_returns_empty():
    store = await _line_graph()
    assert await store.traverse(["zzz"], depth=2) == set()


async def test_link_chunk_and_chunks_for():
    store = InMemoryGraphStore()
    await store.add_entity(Entity(name="Acme"))
    c1, c2 = _chunk("c1"), _chunk("c2")
    await store.link_chunk("acme", c1)
    await store.link_chunk("acme", c2)
    await store.link_chunk("acme", c1)  # dedup by chunk_id
    chunks = await store.chunks_for("acme")
    assert {c.chunk_id for c in chunks} == {"c1", "c2"}


async def test_neighbors_both_directions():
    store = InMemoryGraphStore()
    await store.add_relation(Relation(subject="a", predicate="rel", object="b"))
    assert len(await store.neighbors("a")) == 1
    assert len(await store.neighbors("b")) == 1  # edge stored on both endpoints


async def test_networkx_store_mirrors_graph():
    nx = __import__("importlib").util.find_spec("networkx")
    if nx is None:  # optional 'graph' extra not installed
        return
    store = NetworkxGraphStore()
    await store.add_relation(Relation(subject="a", predicate="acquired", object="b"))
    g = store.as_networkx()
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    # Traversal still works via the inherited in-memory adjacency.
    assert await store.traverse(["a"], depth=1) == {"a", "b"}
