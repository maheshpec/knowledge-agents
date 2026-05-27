"""Knowledge-graph layer for the GraphRAG route (SPEC §7.7).

Re-exports the contracts, store implementations, extractors, and builder so
callers can ``from knowledge_index.graph import EntityGraphBuilder,
InMemoryGraphStore, HeuristicExtractor``.
"""

from knowledge_index.graph.base import (
    Entity,
    EntityExtractor,
    GraphBuilder,
    GraphStore,
    Relation,
    RelationExtractor,
    Triple,
    normalize_entity,
)
from knowledge_index.graph.builder import COOCCURS, EntityGraphBuilder
from knowledge_index.graph.extraction import HeuristicExtractor, LLMExtractor
from knowledge_index.graph.store import (
    InMemoryGraphStore,
    Neo4jGraphStore,
    NetworkxGraphStore,
)

__all__ = [
    "Entity",
    "Relation",
    "Triple",
    "normalize_entity",
    "EntityExtractor",
    "RelationExtractor",
    "GraphStore",
    "GraphBuilder",
    "EntityGraphBuilder",
    "COOCCURS",
    "HeuristicExtractor",
    "LLMExtractor",
    "InMemoryGraphStore",
    "NetworkxGraphStore",
    "Neo4jGraphStore",
]
