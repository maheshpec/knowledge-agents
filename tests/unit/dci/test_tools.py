"""Tests for the DCI tool layer (SPEC §15.1).

Verify each tool is a sandbox-compatible callable, that args are validated, and
that ACLs and per-tool ceilings are enforced inside the tool boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.errors import KnowledgeAgentError
from common.schemas import Chunk
from knowledge_index.dci import (
    CorpusDescribeTool,
    CorpusGlobTool,
    CorpusGrepTool,
    CorpusLsTool,
    CorpusNeighborsTool,
    CorpusReadTool,
    InMemoryCorpusStore,
    make_dci_tools,
)
from knowledge_index.graph.base import Entity, Relation
from knowledge_index.graph.store import InMemoryGraphStore


def _chunk(doc_id, chunk_id, text, *, acl=None, source="docs"):
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        acl=acl or [],
        metadata={"collection": "main", "source": source, "type": "md"},
    )


@pytest.fixture
def store() -> InMemoryCorpusStore:
    s = InMemoryCorpusStore()
    s.add_chunks(
        [
            _chunk("doc-a", "doc-a:0", "Alpha line\nKeyword foo bar"),
            _chunk("doc-b", "doc-b:0", "Beta only"),
            _chunk("doc-s", "doc-s:0", "Secret keyword foo", acl=["team-x"]),
        ]
    )
    return s


# --- contract: every DCI tool is a sandbox Tool ----


def test_every_tool_has_name_and_no_network(store: InMemoryCorpusStore):
    tools = make_dci_tools(store, InMemoryGraphStore())
    expected = {
        "corpus_grep",
        "corpus_glob",
        "corpus_ls",
        "corpus_read",
        "corpus_describe",
        "corpus_neighbors",
    }
    assert set(tools) == expected
    for t in tools.values():
        assert isinstance(t.name, str) and t.name
        assert t.network_required is False


def test_make_dci_tools_omits_neighbors_without_graph(store: InMemoryCorpusStore):
    tools = make_dci_tools(store)
    assert "corpus_neighbors" not in tools
    assert "corpus_grep" in tools


# --- corpus_grep ----


async def test_grep_tool_propagates_principals(store: InMemoryCorpusStore, tmp_path: Path):
    tool = CorpusGrepTool(store)
    public = await tool({"pattern": "[Kk]eyword"}, workdir=tmp_path)
    assert {h.doc_id for h in public} == {"doc-a"}
    private = await tool({"pattern": "[Kk]eyword", "user_principals": ["team-x"]}, workdir=tmp_path)
    assert {h.doc_id for h in private} == {"doc-a", "doc-s"}


async def test_grep_tool_caps_max_hits_at_ceiling(store: InMemoryCorpusStore, tmp_path: Path):
    tool = CorpusGrepTool(store, default_max_hits=1)
    hits = await tool({"pattern": "foo", "max_hits": 99}, workdir=tmp_path)
    assert len(hits) == 1


async def test_grep_tool_requires_pattern(store: InMemoryCorpusStore, tmp_path: Path):
    tool = CorpusGrepTool(store)
    with pytest.raises(KnowledgeAgentError, match="pattern"):
        await tool({}, workdir=tmp_path)


# --- corpus_glob ----


async def test_glob_tool_caps_limit(store: InMemoryCorpusStore, tmp_path: Path):
    tool = CorpusGlobTool(store, default_limit=1)
    refs = await tool({"limit": 100}, workdir=tmp_path)
    assert len(refs) == 1


async def test_glob_tool_validates_types(store: InMemoryCorpusStore, tmp_path: Path):
    tool = CorpusGlobTool(store)
    with pytest.raises(KnowledgeAgentError, match="types"):
        await tool({"types": "not-a-list"}, workdir=tmp_path)


# --- corpus_ls ----


async def test_ls_tool_defaults_to_root(store: InMemoryCorpusStore, tmp_path: Path):
    tool = CorpusLsTool(store)
    listing = await tool({}, workdir=tmp_path)
    assert listing.path == "/"
    assert listing.entries


# --- corpus_read ----


async def test_read_tool_requires_doc_id(store: InMemoryCorpusStore, tmp_path: Path):
    tool = CorpusReadTool(store)
    with pytest.raises(KnowledgeAgentError, match="doc_id"):
        await tool({}, workdir=tmp_path)


async def test_read_tool_caps_max_bytes(store: InMemoryCorpusStore, tmp_path: Path):
    tool = CorpusReadTool(store, default_max_bytes=4)
    slc = await tool({"doc_id": "doc-a", "max_bytes": 9999}, workdir=tmp_path)
    assert len(slc.content) == 4
    assert slc.truncated is True


# --- corpus_describe ----


async def test_describe_tool_returns_metadata(store: InMemoryCorpusStore, tmp_path: Path):
    tool = CorpusDescribeTool(store)
    md = await tool({"doc_id": "doc-a"}, workdir=tmp_path)
    assert md.doc_id == "doc-a"
    assert md.length > 0


async def test_describe_tool_hides_private_doc_without_principals(
    store: InMemoryCorpusStore, tmp_path: Path
):
    tool = CorpusDescribeTool(store)
    md = await tool({"doc_id": "doc-s"}, workdir=tmp_path)
    assert md.metadata.get("hidden") is True


# --- corpus_neighbors ----


@pytest.fixture
async def graph_with_chunks() -> tuple[InMemoryGraphStore, Chunk, Chunk, Chunk]:
    store = InMemoryGraphStore()
    a = Chunk(chunk_id="c-seed", doc_id="doc-x", text="seed text")
    b = Chunk(chunk_id="c-near", doc_id="doc-y", text="near text")
    c = Chunk(chunk_id="c-far", doc_id="doc-z", text="far text")
    await store.add_entity(Entity(name="alpha", key="alpha"))
    await store.add_entity(Entity(name="beta", key="beta"))
    await store.add_entity(Entity(name="gamma", key="gamma"))
    await store.add_relation(Relation(subject="alpha", predicate="r", object="beta"))
    await store.add_relation(Relation(subject="beta", predicate="r", object="gamma"))
    await store.link_chunk("alpha", a)
    await store.link_chunk("beta", b)
    await store.link_chunk("gamma", c)
    return store, a, b, c


async def test_neighbors_returns_one_hop_chunks(graph_with_chunks, tmp_path: Path):
    store, seed, near, far = graph_with_chunks
    tool = CorpusNeighborsTool(store)
    refs = await tool({"chunk_id": seed.chunk_id, "hops": 1}, workdir=tmp_path)
    ids = {r.chunk_id for r in refs}
    assert near.chunk_id in ids
    assert far.chunk_id not in ids  # 2 hops away
    assert seed.chunk_id not in ids  # don't return the seed itself


async def test_neighbors_extends_to_two_hops(graph_with_chunks, tmp_path: Path):
    store, seed, near, far = graph_with_chunks
    tool = CorpusNeighborsTool(store)
    refs = await tool({"chunk_id": seed.chunk_id, "hops": 2}, workdir=tmp_path)
    by_id = {r.chunk_id: r for r in refs}
    assert near.chunk_id in by_id and far.chunk_id in by_id
    assert by_id[near.chunk_id].hops == 1
    assert by_id[far.chunk_id].hops == 2


async def test_neighbors_respects_acl(graph_with_chunks, tmp_path: Path):
    store, seed, _near, _far = graph_with_chunks
    private = Chunk(chunk_id="c-private", doc_id="doc-p", text="hidden", acl=["team-x"])
    await store.link_chunk("beta", private)
    tool = CorpusNeighborsTool(store)
    refs = await tool({"chunk_id": seed.chunk_id, "hops": 1}, workdir=tmp_path)
    assert "c-private" not in {r.chunk_id for r in refs}
    refs = await tool(
        {"chunk_id": seed.chunk_id, "hops": 1, "user_principals": ["team-x"]},
        workdir=tmp_path,
    )
    assert "c-private" in {r.chunk_id for r in refs}


async def test_neighbors_empty_when_chunk_unknown(graph_with_chunks, tmp_path: Path):
    store, *_ = graph_with_chunks
    tool = CorpusNeighborsTool(store)
    refs = await tool({"chunk_id": "no-such-chunk"}, workdir=tmp_path)
    assert refs == []


async def test_neighbors_caps_max_hops(graph_with_chunks, tmp_path: Path):
    store, seed, near, far = graph_with_chunks
    tool = CorpusNeighborsTool(store, max_hops=1)
    refs = await tool({"chunk_id": seed.chunk_id, "hops": 99}, workdir=tmp_path)
    ids = {r.chunk_id for r in refs}
    assert far.chunk_id not in ids  # max_hops clamps the walk


# --- args validation ----


def test_principals_arg_must_be_list(store: InMemoryCorpusStore):
    from knowledge_index.dci.tools import _principals

    with pytest.raises(KnowledgeAgentError):
        _principals({"user_principals": "alice"})  # must be a list
