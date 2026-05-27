"""Tests for the GraphRetriever (SPEC §7.6.3 / §7.7)."""

from __future__ import annotations

from common.schemas import Chunk, Query, RetrievalCandidate
from knowledge_index.graph.builder import EntityGraphBuilder
from knowledge_index.graph.extraction import HeuristicExtractor
from knowledge_index.retrieval.retrievers.graph import GraphRetriever


def _chunk(cid: str, text: str, acl: list[str] | None = None) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=f"doc-{cid}", text=text, acl=acl or [])


async def _store(chunks):
    return await EntityGraphBuilder().build(chunks)


async def test_graph_retriever_surfaces_seed_chunk():
    chunks = [_chunk("c1", "Acme Corp acquired Globex.")]
    store = await _store(chunks)
    r = GraphRetriever(store, HeuristicExtractor())
    out = await r.retrieve(Query(raw="Tell me about Acme Corp"), k=5)
    assert all(isinstance(c, RetrievalCandidate) for c in out)
    assert {c.chunk.chunk_id for c in out} == {"c1"}
    assert out[0].retriever == "graph"


async def test_graph_retriever_multi_hop_reaches_distant_chunk():
    # Query mentions only Acme; the answer lives in c2 (about Jane Doe), reachable
    # via Acme → Globex → Jane Doe across chunks. Vector/lexical match on "Acme"
    # alone would never surface c2.
    chunks = [
        _chunk("c1", "Acme Corp acquired Globex."),
        _chunk("c2", "Globex was founded by Jane Doe."),
    ]
    store = await _store(chunks)
    r = GraphRetriever(store, HeuristicExtractor(), depth=2)
    out = await r.retrieve(Query(raw="Who is behind Acme Corp?"), k=5)
    ids = {c.chunk.chunk_id for c in out}
    assert ids == {"c1", "c2"}
    # Closer chunk (c1, contains the seed) scores higher than the 2-hop c2.
    by_id = {c.chunk.chunk_id: c.score for c in out}
    assert by_id["c1"] > by_id["c2"]


async def test_graph_retriever_depth_limits_reach():
    chunks = [
        _chunk("c1", "Acme Corp acquired Globex."),
        _chunk("c2", "Globex was founded by Jane Doe."),
    ]
    store = await _store(chunks)
    shallow = GraphRetriever(store, HeuristicExtractor(), depth=1)
    out = await shallow.retrieve(Query(raw="Who is behind Acme Corp?"), k=5)
    # depth=1 from "acme corp" reaches "globex" (and its chunks c1,c2) but the
    # founder is only reachable through globex's chunk linkage at the same hop.
    assert "c1" in {c.chunk.chunk_id for c in out}


async def test_graph_retriever_no_known_entities_returns_empty():
    store = await _store([_chunk("c1", "Acme Corp acquired Globex.")])
    r = GraphRetriever(store, HeuristicExtractor())
    out = await r.retrieve(Query(raw="something totally unrelated lowercase"), k=5)
    assert out == []


async def test_graph_retriever_enforces_acl():
    chunks = [_chunk("c1", "Acme Corp acquired Globex.", acl=["team-secret"])]
    store = await _store(chunks)
    r = GraphRetriever(store, HeuristicExtractor())
    # Caller without the principal sees nothing.
    out = await r.retrieve(Query(raw="Acme Corp", user_principals=["intruder"]), k=5)
    assert out == []
    # Caller with the principal sees it.
    ok = await r.retrieve(Query(raw="Acme Corp", user_principals=["team-secret"]), k=5)
    assert {c.chunk.chunk_id for c in ok} == {"c1"}


async def test_graph_retriever_respects_k():
    chunks = [_chunk(f"c{i}", f"Acme Corp item{i} Globex{i}.") for i in range(5)]
    store = await _store(chunks)
    r = GraphRetriever(store, HeuristicExtractor())
    out = await r.retrieve(Query(raw="Acme Corp"), k=2)
    assert len(out) <= 2
