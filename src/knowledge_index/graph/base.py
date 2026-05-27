"""Knowledge-graph contracts: entities, relations, store/extractor/builder (SPEC §7.7).

The GraphRAG route is built on a small, store-agnostic contract so the dev store
(`networkx`/in-memory) and the prod store (Neo4j) are interchangeable behind one
:class:`GraphStore` protocol. Ingestion populates the graph via a
:class:`GraphBuilder`; retrieval traverses it via
:class:`~knowledge_index.retrieval.retrievers.graph.GraphRetriever`.

The graph is *entity-centric*: nodes are extracted entities, edges are relations
between them, and every entity node carries the set of ``chunk_id``s that mention
it. Multi-hop traversal over relations therefore surfaces chunks that are
*topically connected* even when they share no query terms — the property that
lets GraphRAG beat the vector route on relational queries (SPEC §10 Phase 3).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from common.schemas import Chunk


def normalize_entity(name: str) -> str:
    """Canonical key for an entity surface form: trimmed, casefolded, collapsed.

    Entity resolution in the dev store is intentionally simple — surface forms
    that normalize to the same key are the same node. Prod (Neo4j) can layer a
    richer resolver on top; callers only depend on this being deterministic.
    """
    return " ".join(name.split()).casefold()


class Entity(BaseModel):
    """A node in the knowledge graph (an extracted domain term/named entity)."""

    name: str  # canonical surface form (display)
    type: str = "concept"  # NER-style label: person, org, concept, ...
    key: str = ""  # normalized identity key; filled from ``name`` if blank

    def model_post_init(self, __context: object) -> None:
        if not self.key:
            object.__setattr__(self, "key", normalize_entity(self.name))


class Relation(BaseModel):
    """A directed, typed edge between two entities (a subject-predicate-object triple)."""

    subject: str  # entity key
    predicate: str  # relation label, e.g. "acquired", "founded_by"
    object: str  # entity key
    chunk_id: str | None = None  # provenance: chunk the triple was extracted from


class Triple(BaseModel):
    """An extractor's raw output before key-normalization: (subject, predicate, object)."""

    subject: str
    predicate: str
    object: str


@runtime_checkable
class EntityExtractor(Protocol):
    """Pull entities out of free text (NER + LLM-based domain-term extraction)."""

    async def extract_entities(self, text: str) -> list[Entity]: ...


@runtime_checkable
class RelationExtractor(Protocol):
    """Pull subject-predicate-object triples out of free text (constrained schema)."""

    async def extract_triples(self, text: str) -> list[Triple]: ...


@runtime_checkable
class GraphStore(Protocol):
    """Persistence + traversal surface for the KG (Neo4j or in-memory/networkx).

    Implementations must be safe to call concurrently for reads. Writes happen
    only during the build phase, single-threaded per the ingestion pipeline.
    """

    async def add_entity(self, entity: Entity) -> None: ...

    async def add_relation(self, relation: Relation) -> None: ...

    async def link_chunk(self, entity_key: str, chunk: Chunk) -> None:
        """Record that ``chunk`` mentions the entity ``entity_key``."""
        ...

    async def neighbors(self, entity_key: str) -> list[Relation]:
        """Relations incident to ``entity_key`` (both directions)."""
        ...

    async def traverse(self, seeds: list[str], depth: int) -> set[str]:
        """BFS over relations from ``seeds``; return all entity keys within ``depth`` hops."""
        ...

    async def chunks_for(self, entity_key: str) -> list[Chunk]:
        """Chunks that mention ``entity_key`` (empty if unknown)."""
        ...

    async def has_entity(self, entity_key: str) -> bool: ...


@runtime_checkable
class GraphBuilder(Protocol):
    """Populate a :class:`GraphStore` from chunks (SPEC §7.7).

    Takes ``Chunk``s rather than ``ParsedDoc``s (a deliberate refinement of the
    SPEC sketch): retrieval must surface chunk-linked entity nodes, so the build
    step needs chunk identity, not just document text.
    """

    async def build(self, chunks: list[Chunk]) -> GraphStore: ...


# A scored entity hit produced while ranking a traversal frontier.
class GraphHit(BaseModel):
    """An entity reached during traversal, with its hop distance from the seeds."""

    entity_key: str
    hops: int = Field(ge=0)


__all__ = [
    "normalize_entity",
    "Entity",
    "Relation",
    "Triple",
    "EntityExtractor",
    "RelationExtractor",
    "GraphStore",
    "GraphBuilder",
    "GraphHit",
]
