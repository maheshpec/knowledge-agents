"""Build the knowledge graph from chunks (SPEC §7.7).

For each chunk the builder:

1. extracts entities and links them to the chunk (entity → chunk_id),
2. extracts relation triples and adds them as typed edges,
3. adds *co-occurrence* edges between entities that appear in the same chunk.

Co-occurrence edges are what make multi-hop traversal useful even when the
text states no explicit relation: two entities mentioned together are one hop
apart, so a query seeded on one can reach chunks about the other. Explicit
triples (acquired, founded_by, …) carry the typed predicate on top.

Extraction runs concurrently across chunks (bounded) and the store is written
single-threaded, matching the ingestion pipeline's contract.
"""

from __future__ import annotations

import asyncio
import itertools

from common.schemas import Chunk
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from knowledge_index.graph.base import (
    Entity,
    EntityExtractor,
    Relation,
    RelationExtractor,
    normalize_entity,
)
from knowledge_index.graph.extraction import HeuristicExtractor
from knowledge_index.graph.store import InMemoryGraphStore

_log = get_logger("knowledge_index.graph.builder")

# Predicate used for the implicit "appeared in the same chunk" edge.
COOCCURS = "co_occurs"


class EntityGraphBuilder:
    """Populate a :class:`GraphStore` from chunks via entity/relation extraction."""

    def __init__(
        self,
        *,
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
        max_concurrency: int = 8,
        cooccurrence: bool = True,
    ) -> None:
        default = HeuristicExtractor()
        self._entities = entity_extractor or default
        self._relations = relation_extractor or default
        self._max_concurrency = max_concurrency
        self._cooccurrence = cooccurrence

    @traced(span_name="graph.build")
    async def build(self, chunks: list[Chunk]) -> InMemoryGraphStore:
        store = InMemoryGraphStore()
        sem = asyncio.Semaphore(self._max_concurrency)

        async def extract(chunk: Chunk) -> tuple[Chunk, list[Entity], list[Relation]]:
            text = f"{chunk.context}\n{chunk.text}" if chunk.context else chunk.text
            async with sem:
                ents = await self._entities.extract_entities(text)
                triples = await self._relations.extract_triples(text)
            rels = [
                Relation(
                    subject=normalize_entity(t.subject),
                    predicate=t.predicate,
                    object=normalize_entity(t.object),
                    chunk_id=chunk.chunk_id,
                )
                for t in triples
            ]
            return chunk, ents, rels

        extracted = await asyncio.gather(*(extract(c) for c in chunks))

        # Single-threaded write phase: entities, chunk links, relations, co-occ edges.
        for chunk, ents, rels in extracted:
            keys = [e.key for e in ents]
            for ent in ents:
                await store.add_entity(ent)
                await store.link_chunk(ent.key, chunk)
            for rel in rels:
                # Ensure relation endpoints are linked to this chunk too, so a
                # typed-edge hop surfaces the originating chunk.
                await store.add_relation(rel)
                await store.link_chunk(rel.subject, chunk)
                await store.link_chunk(rel.object, chunk)
            if self._cooccurrence:
                for a, b in itertools.combinations(sorted(set(keys)), 2):
                    await store.add_relation(
                        Relation(subject=a, predicate=COOCCURS, object=b, chunk_id=chunk.chunk_id)
                    )

        _log.info(
            "graph.built",
            chunks=len(chunks),
            entities=store.entity_count(),
            relations=store.relation_count(),
        )
        return store


__all__ = ["EntityGraphBuilder", "COOCCURS"]
