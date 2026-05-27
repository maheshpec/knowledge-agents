"""GraphRAG retriever: multi-hop traversal on the KG (SPEC §7.6.3 / §7.7).

Retrieval flow:

1. Extract entities from the query (same extractor used at build time).
2. Seed traversal on the entity keys present in the graph, then BFS outward to
   ``depth`` hops, recording each reached entity's hop distance.
3. Surface the chunks linked to reached entities, scoring each chunk by graph
   proximity — closer entities and more distinct linking entities score higher.

This is what beats the vector route on *relational* queries: a chunk that shares
no terms with the query still surfaces if it sits on a relation path from a query
entity (e.g. "who founded the company that acquired X" reaches the founder chunk
via X → acquirer → founder). The retriever is store-agnostic — it traverses via
the :class:`~knowledge_index.graph.base.GraphStore` ``neighbors`` API, so the
in-memory dev store and Neo4j behave identically.
"""

from __future__ import annotations

from collections import deque

from common.schemas import Chunk, Query, RetrievalCandidate
from harness.observability.tracing import traced
from knowledge_index.graph.base import EntityExtractor, GraphStore, normalize_entity


class GraphRetriever:
    """Multi-hop KG retriever, exposed as ``strategy='graph'`` (SPEC §7.6.1 line 787)."""

    name = "graph"

    def __init__(
        self,
        store: GraphStore,
        extractor: EntityExtractor,
        *,
        depth: int = 2,
    ) -> None:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self._store = store
        self._extractor = extractor
        self._depth = depth

    async def _seed_keys(self, query: Query) -> list[str]:
        """Entity keys from the query that actually exist in the graph."""
        entities = await self._extractor.extract_entities(query.raw)
        keys: list[str] = []
        seen: set[str] = set()
        for ent in entities:
            key = ent.key or normalize_entity(ent.name)
            if key in seen:
                continue
            seen.add(key)
            if await self._store.has_entity(key):
                keys.append(key)
        return keys

    async def _bfs(self, seeds: list[str]) -> dict[str, int]:
        """BFS from seeds via ``neighbors``; return entity_key -> min hop distance."""
        hops: dict[str, int] = {s: 0 for s in seeds}
        frontier: deque[str] = deque(seeds)
        while frontier:
            key = frontier.popleft()
            d = hops[key]
            if d >= self._depth:
                continue
            for rel in await self._store.neighbors(key):
                for other in (rel.subject, rel.object):
                    if other not in hops:
                        hops[other] = d + 1
                        frontier.append(other)
        return hops

    @traced(span_name="retrieval.graph")
    async def retrieve(self, query: Query, k: int) -> list[RetrievalCandidate]:
        seeds = await self._seed_keys(query)
        if not seeds:
            return []
        hops = await self._bfs(seeds)

        # Accumulate per-chunk score: each linking entity contributes 1/(1+hops),
        # so chunks reachable via closer / multiple entities rank higher. Respect
        # ACL: a chunk is visible iff its acl intersects the caller's principals
        # (empty acl => public), matching the index-side contract (SPEC §7.5).
        principals = set(query.user_principals)
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for key, hop in hops.items():
            weight = 1.0 / (1.0 + hop)
            for chunk in await self._store.chunks_for(key):
                if chunk.acl and not (set(chunk.acl) & principals):
                    continue
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight
                chunks[chunk.chunk_id] = chunk

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            RetrievalCandidate(chunk=chunks[cid], score=score, retriever=self.name, rank=i)
            for i, (cid, score) in enumerate(ranked, start=1)
        ]


__all__ = ["GraphRetriever"]
