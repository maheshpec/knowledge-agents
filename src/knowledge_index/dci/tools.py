"""DCI tool implementations (SPEC §15.1).

Each tool satisfies the :class:`harness.sandbox.Tool` protocol — a name, a
``network_required`` flag, and an ``async __call__(args, workdir)`` — so the
orchestrator's :class:`SandboxedToolExecutor` (SPEC §6.7) can run them under a
policy without special-casing. Tools are constructed with their backing store
(:class:`CorpusStore` or :class:`GraphStore`) and a small per-tool ceiling
(``max_hits``, ``max_bytes``); each invocation accepts ``user_principals`` in
its args so ACLs are enforced inside the store, not after retrieval.

The :func:`make_dci_tools` factory builds the six tools as a name→tool dict
ready to drop into ``OrchestratorDeps.tools`` (SPEC §6.1, §15.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from common.errors import KnowledgeAgentError
from knowledge_index.dci.base import (
    ChunkRef,
    CorpusStore,
    DirectoryListing,
    DocMetadata,
    DocRef,
    DocSlice,
    GrepHit,
)
from knowledge_index.graph.base import GraphStore


class DCITool(Protocol):
    """The shape every DCI tool exports (a sandbox :class:`Tool`)."""

    name: str
    network_required: bool

    async def __call__(self, args: dict[str, Any], *, workdir: Path) -> Any: ...


def _principals(args: dict[str, Any]) -> list[str] | None:
    """Pull ``user_principals`` from tool args; ``None`` => no principals."""
    val = args.get("user_principals")
    if val is None:
        return None
    if not isinstance(val, list):
        raise KnowledgeAgentError("user_principals must be a list of strings")
    return [str(p) for p in val]


def _require(args: dict[str, Any], key: str) -> Any:
    if key not in args:
        raise KnowledgeAgentError(f"missing required arg: {key!r}")
    return args[key]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class CorpusGrepTool:
    """ACL-filtered regex over raw doc text (SPEC §15.1)."""

    name = "corpus_grep"
    network_required = False

    def __init__(
        self,
        store: CorpusStore,
        *,
        default_max_hits: int = 50,
        default_context_lines: int = 2,
    ) -> None:
        self._store = store
        self._default_max_hits = default_max_hits
        self._default_context_lines = default_context_lines

    async def __call__(self, args: dict[str, Any], *, workdir: Path) -> list[GrepHit]:
        pattern = _require(args, "pattern")
        glob = args.get("glob", "**/*")
        regex = bool(args.get("regex", True))
        max_hits = min(int(args.get("max_hits", self._default_max_hits)), self._default_max_hits)
        context_lines = int(args.get("context_lines", self._default_context_lines))
        return await self._store.grep(
            str(pattern),
            glob=str(glob),
            regex=regex,
            max_hits=max(0, max_hits),
            context_lines=max(0, context_lines),
            principals=_principals(args),
        )


class CorpusGlobTool:
    """Path-pattern listing within the ACL slice (SPEC §15.1)."""

    name = "corpus_glob"
    network_required = False

    def __init__(self, store: CorpusStore, *, default_limit: int = 200) -> None:
        self._store = store
        self._default_limit = default_limit

    async def __call__(self, args: dict[str, Any], *, workdir: Path) -> list[DocRef]:
        pattern = args.get("pattern", "**/*")
        types = args.get("types")
        if types is not None and not isinstance(types, list):
            raise KnowledgeAgentError("types must be a list of strings")
        limit = min(int(args.get("limit", self._default_limit)), self._default_limit)
        return await self._store.glob(
            str(pattern),
            types=[str(t) for t in types] if types else None,
            limit=max(0, limit),
            principals=_principals(args),
        )


class CorpusLsTool:
    """Browse the logical tree (collection → source → doc) (SPEC §15.1)."""

    name = "corpus_ls"
    network_required = False

    def __init__(self, store: CorpusStore) -> None:
        self._store = store

    async def __call__(self, args: dict[str, Any], *, workdir: Path) -> DirectoryListing:
        path = args.get("path", "/")
        return await self._store.ls(str(path), principals=_principals(args))


class CorpusReadTool:
    """Full-or-windowed doc read with a citation (SPEC §15.1)."""

    name = "corpus_read"
    network_required = False

    def __init__(self, store: CorpusStore, *, default_max_bytes: int = 50_000) -> None:
        self._store = store
        self._default_max_bytes = default_max_bytes

    async def __call__(self, args: dict[str, Any], *, workdir: Path) -> DocSlice:
        doc_id = _require(args, "doc_id")
        start_line = int(args.get("start_line", 1))
        end_line = args.get("end_line")
        if end_line is not None:
            end_line = int(end_line)
        # Clamp ``max_bytes`` to the per-tool ceiling so a caller can shrink the
        # window but not exceed the policy budget.
        max_bytes = min(
            int(args.get("max_bytes", self._default_max_bytes)), self._default_max_bytes
        )
        return await self._store.read(
            str(doc_id),
            start_line=start_line,
            end_line=end_line,
            max_bytes=max(0, max_bytes),
            principals=_principals(args),
        )


class CorpusDescribeTool:
    """Doc metadata: title / source / authors / length / ACL / ingest time."""

    name = "corpus_describe"
    network_required = False

    def __init__(self, store: CorpusStore) -> None:
        self._store = store

    async def __call__(self, args: dict[str, Any], *, workdir: Path) -> DocMetadata:
        doc_id = _require(args, "doc_id")
        return await self._store.describe(str(doc_id), principals=_principals(args))


class CorpusNeighborsTool:
    """KG walk from a chunk (reuses Phase 3M GraphRetriever store) (SPEC §15.1)."""

    name = "corpus_neighbors"
    network_required = False

    def __init__(
        self,
        graph_store: GraphStore,
        *,
        default_hops: int = 1,
        max_hops: int = 3,
        max_results: int = 50,
    ) -> None:
        self._store = graph_store
        self._default_hops = default_hops
        self._max_hops = max_hops
        self._max_results = max_results

    async def __call__(self, args: dict[str, Any], *, workdir: Path) -> list[ChunkRef]:
        chunk_id = _require(args, "chunk_id")
        hops = min(max(0, int(args.get("hops", self._default_hops))), self._max_hops)
        principals = _principals(args)

        # The KG indexes chunks by *entity* — to walk from a chunk we first
        # find the entities the KG links to this chunk_id (seeds), then BFS
        # outward via the protocol's ``neighbors`` API, recording each reached
        # entity's hop distance. This mirrors GraphRetriever (Phase 3M) so a
        # store that satisfies the protocol works for both.
        seed_entities = await _seed_entities_for_chunk(self._store, str(chunk_id))
        if not seed_entities:
            return []
        hop_by_entity = await _bfs_hops(self._store, seed_entities, hops)

        # Collect chunks linked to reached entities, ACL-filtered, dedup'd.
        # When the same chunk appears via multiple entities, keep the lowest
        # hop count (most direct connection).
        from collections import OrderedDict

        out: OrderedDict[str, ChunkRef] = OrderedDict()
        for entity_key, hop_dist in hop_by_entity.items():
            for chunk in await self._store.chunks_for(entity_key):
                if chunk.chunk_id == str(chunk_id):
                    continue
                if chunk.acl and (principals is None or not (set(chunk.acl) & set(principals))):
                    continue
                existing = out.get(chunk.chunk_id)
                if existing is not None and existing.hops <= hop_dist:
                    continue
                out[chunk.chunk_id] = ChunkRef(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    hops=hop_dist,
                    citation=_chunk_citation(chunk),
                )
                if len(out) >= self._max_results:
                    break
            if len(out) >= self._max_results:
                break
        # Stable ordering: hop distance ascending, then insertion order.
        return sorted(out.values(), key=lambda r: r.hops)


# ---------------------------------------------------------------------------
# Helpers + factory
# ---------------------------------------------------------------------------


async def _bfs_hops(store: GraphStore, seeds: list[str], depth: int) -> dict[str, int]:
    """BFS from ``seeds`` via ``neighbors``; return entity_key -> min hops."""
    from collections import deque

    hops: dict[str, int] = {s: 0 for s in seeds}
    frontier: deque[str] = deque(seeds)
    while frontier:
        key = frontier.popleft()
        d = hops[key]
        if d >= depth:
            continue
        for rel in await store.neighbors(key):
            for other in (rel.subject, rel.object):
                if other not in hops:
                    hops[other] = d + 1
                    frontier.append(other)
    return hops


async def _seed_entities_for_chunk(store: GraphStore, chunk_id: str) -> list[str]:
    """Find entities the KG links to ``chunk_id`` (seeds for neighbor traversal).

    The :class:`GraphStore` protocol gives us ``chunks_for(entity_key)`` and a
    full-graph traversal, but no reverse index. For the in-memory dev store we
    walk its private ``_chunks`` mapping when present; otherwise we fall back
    to a generic scan over the public protocol. Either way, the result is the
    set of entity keys whose linked chunks include ``chunk_id``.
    """
    private_chunks = getattr(store, "_chunks", None)
    if isinstance(private_chunks, dict):
        return [key for key, chunks in private_chunks.items() if chunk_id in chunks]

    # Fallback: enumerate via the public surface. There is no entity-listing
    # method on the protocol, so this path is best-effort and intended only
    # for stores that surface their entity set some other way.
    entities = getattr(store, "_entities", None)
    if not isinstance(entities, dict):
        return []
    seeds: list[str] = []
    for key in entities:
        chunks = await store.chunks_for(key)
        if any(c.chunk_id == chunk_id for c in chunks):
            seeds.append(key)
    return seeds


def _chunk_citation(chunk: Any) -> Any:
    """Build a :class:`Source` pointing at ``chunk``."""
    from common.schemas import Source

    return Source(
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        parent_id=getattr(chunk, "parent_id", None),
        metadata=dict(getattr(chunk, "metadata", {}) or {}),
    )


def make_dci_tools(
    corpus_store: CorpusStore,
    graph_store: GraphStore | None = None,
    *,
    max_hits: int = 50,
    max_bytes: int = 50_000,
    default_limit: int = 200,
    default_hops: int = 1,
) -> dict[str, DCITool]:
    """Build the DCI tool set as a ``{name: tool}`` map for the orchestrator.

    Pass the resulting dict as ``OrchestratorDeps.tools``. ``graph_store`` is
    optional — without it, ``corpus_neighbors`` is omitted. The keyword caps
    set per-tool ceilings; callers can still shrink them per invocation in the
    tool args but cannot exceed them.
    """
    tools: dict[str, DCITool] = {
        "corpus_grep": CorpusGrepTool(corpus_store, default_max_hits=max_hits),
        "corpus_glob": CorpusGlobTool(corpus_store, default_limit=default_limit),
        "corpus_ls": CorpusLsTool(corpus_store),
        "corpus_read": CorpusReadTool(corpus_store, default_max_bytes=max_bytes),
        "corpus_describe": CorpusDescribeTool(corpus_store),
    }
    if graph_store is not None:
        tools["corpus_neighbors"] = CorpusNeighborsTool(graph_store, default_hops=default_hops)
    return tools


__all__ = [
    "DCITool",
    "CorpusGrepTool",
    "CorpusGlobTool",
    "CorpusLsTool",
    "CorpusReadTool",
    "CorpusDescribeTool",
    "CorpusNeighborsTool",
    "make_dci_tools",
]
