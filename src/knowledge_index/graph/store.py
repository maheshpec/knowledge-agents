"""Graph store implementations (SPEC §3 line 92 / §7.7 / §12).

:class:`InMemoryGraphStore` is the default *dev* store: pure-stdlib adjacency
lists, no third-party dependency, deterministic traversal — so unit tests and
the eval harness run on the lean core install. :class:`NetworkxGraphStore` wraps
it with a ``networkx.MultiDiGraph`` view for callers who want graph algorithms,
and :class:`Neo4jGraphStore` is the prod target; both import their backend
lazily (optional ``graph`` extra) so importing this module is always cheap.

All three satisfy the :class:`~knowledge_index.graph.base.GraphStore` protocol.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from common.schemas import Chunk
from knowledge_index.graph.base import Entity, Relation, normalize_entity


class InMemoryGraphStore:
    """Adjacency-list KG store with chunk linkage and bounded BFS traversal."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        # adjacency: entity_key -> relations where it is the subject OR object
        self._adj: dict[str, list[Relation]] = {}
        # entity_key -> chunk_id -> Chunk (dedup by chunk_id)
        self._chunks: dict[str, dict[str, Chunk]] = {}

    async def add_entity(self, entity: Entity) -> None:
        self._entities.setdefault(entity.key, entity)
        self._adj.setdefault(entity.key, [])
        self._chunks.setdefault(entity.key, {})

    async def add_relation(self, relation: Relation) -> None:
        # Ensure both endpoints exist as nodes (an edge implies its vertices).
        for key in (relation.subject, relation.object):
            if key not in self._entities:
                await self.add_entity(Entity(name=key, key=key))
        # Store on both endpoints so ``neighbors``/``traverse`` is undirected for
        # reachability while ``predicate``/direction stay inspectable on the edge.
        self._adj[relation.subject].append(relation)
        if relation.object != relation.subject:
            self._adj[relation.object].append(relation)

    async def link_chunk(self, entity_key: str, chunk: Chunk) -> None:
        if entity_key not in self._entities:
            await self.add_entity(Entity(name=entity_key, key=entity_key))
        self._chunks[entity_key][chunk.chunk_id] = chunk

    async def neighbors(self, entity_key: str) -> list[Relation]:
        return list(self._adj.get(entity_key, ()))

    async def traverse(self, seeds: list[str], depth: int) -> set[str]:
        """BFS to ``depth`` hops; returns reached entity keys including seeds."""
        reached: set[str] = set()
        frontier: deque[tuple[str, int]] = deque()
        for s in seeds:
            key = normalize_entity(s) if s not in self._entities else s
            if key in self._entities and key not in reached:
                reached.add(key)
                frontier.append((key, 0))
        while frontier:
            key, hop = frontier.popleft()
            if hop >= depth:
                continue
            for rel in self._adj.get(key, ()):
                for other in (rel.subject, rel.object):
                    if other not in reached:
                        reached.add(other)
                        frontier.append((other, hop + 1))
        return reached

    async def chunks_for(self, entity_key: str) -> list[Chunk]:
        return list(self._chunks.get(entity_key, {}).values())

    async def has_entity(self, entity_key: str) -> bool:
        return entity_key in self._entities

    # --- introspection helpers (not part of the protocol; used by tests/tools) ---

    def entity_count(self) -> int:
        return len(self._entities)

    def relation_count(self) -> int:
        # Each relation is stored on up to two adjacency lists; count unique.
        seen: set[tuple[str, str, str]] = set()
        for rels in self._adj.values():
            for r in rels:
                seen.add((r.subject, r.predicate, r.object))
        return len(seen)


class NetworkxGraphStore(InMemoryGraphStore):
    """In-memory store that also maintains a ``networkx.MultiDiGraph`` mirror.

    Inherits all storage/traversal from :class:`InMemoryGraphStore` (so it works
    without the dependency) and additionally exposes :meth:`as_networkx` for
    callers who want centrality/path algorithms. ``networkx`` is imported lazily.
    """

    def __init__(self) -> None:
        super().__init__()
        self._nx: Any = None

    def _graph(self) -> Any:
        if self._nx is None:
            import networkx as nx  # type: ignore  # optional 'graph' extra

            self._nx = nx.MultiDiGraph()
        return self._nx

    async def add_entity(self, entity: Entity) -> None:
        await super().add_entity(entity)
        self._graph().add_node(entity.key, type=entity.type, name=entity.name)

    async def add_relation(self, relation: Relation) -> None:
        await super().add_relation(relation)
        self._graph().add_edge(
            relation.subject,
            relation.object,
            predicate=relation.predicate,
            chunk_id=relation.chunk_id,
        )

    def as_networkx(self) -> Any:
        """Return the underlying ``networkx.MultiDiGraph`` (builds it if needed)."""
        return self._graph()


class Neo4jGraphStore:
    """Prod KG store backed by Neo4j (SPEC §3 line 92). Lazily imports ``neo4j``.

    Implemented against the protocol but kept thin: the dev/eval path uses
    :class:`InMemoryGraphStore`; this exists so the production wiring has a real
    target and the import surface is validated. Requires the optional ``graph``
    extra and a running Neo4j instance.
    """

    def __init__(self, uri: str, auth: tuple[str, str], *, database: str = "neo4j") -> None:
        from neo4j import AsyncGraphDatabase  # type: ignore  # optional 'graph' extra

        self._driver = AsyncGraphDatabase.driver(uri, auth=auth)
        self._db = database
        # Chunk bodies are not stored in Neo4j; we keep them addressable in-process
        # keyed by entity so ``chunks_for`` can rebuild RetrievalCandidates.
        self._chunks: dict[str, dict[str, Chunk]] = {}

    async def add_entity(self, entity: Entity) -> None:
        async with self._driver.session(database=self._db) as s:
            await s.run(
                "MERGE (e:Entity {key:$key}) SET e.name=$name, e.type=$type",
                key=entity.key,
                name=entity.name,
                type=entity.type,
            )

    async def add_relation(self, relation: Relation) -> None:
        async with self._driver.session(database=self._db) as s:
            await s.run(
                "MERGE (a:Entity {key:$s}) MERGE (b:Entity {key:$o}) "
                "MERGE (a)-[r:REL {predicate:$p}]->(b) SET r.chunk_id=$c",
                s=relation.subject,
                o=relation.object,
                p=relation.predicate,
                c=relation.chunk_id,
            )

    async def link_chunk(self, entity_key: str, chunk: Chunk) -> None:
        self._chunks.setdefault(entity_key, {})[chunk.chunk_id] = chunk

    async def neighbors(self, entity_key: str) -> list[Relation]:
        async with self._driver.session(database=self._db) as s:
            res = await s.run(
                "MATCH (a:Entity {key:$k})-[r:REL]-(b:Entity) "
                "RETURN startNode(r).key AS s, r.predicate AS p, endNode(r).key AS o, "
                "r.chunk_id AS c",
                k=entity_key,
            )
            rows = [rec async for rec in res]
        return [Relation(subject=r["s"], predicate=r["p"], object=r["o"], chunk_id=r["c"]) for r in rows]

    async def traverse(self, seeds: list[str], depth: int) -> set[str]:
        async with self._driver.session(database=self._db) as s:
            res = await s.run(
                f"MATCH (a:Entity)-[*0..{int(depth)}]-(b:Entity) "
                "WHERE a.key IN $seeds RETURN collect(DISTINCT b.key) AS keys",
                seeds=[normalize_entity(x) for x in seeds],
            )
            rec = await res.single()
        return set(rec["keys"]) if rec else set()

    async def chunks_for(self, entity_key: str) -> list[Chunk]:
        return list(self._chunks.get(entity_key, {}).values())

    async def has_entity(self, entity_key: str) -> bool:
        async with self._driver.session(database=self._db) as s:
            res = await s.run(
                "MATCH (e:Entity {key:$k}) RETURN count(e) AS n", k=entity_key
            )
            rec = await res.single()
        return bool(rec and rec["n"])

    async def close(self) -> None:
        await self._driver.close()


__all__ = ["InMemoryGraphStore", "NetworkxGraphStore", "Neo4jGraphStore"]
