"""Query routing (SPEC §7.6.1): intent-aware strategy selection.

Re-exports the routing surface so callers can ``from
knowledge_index.retrieval.routers import LLMRouter, RouterPipeline, RouteDecision``.
"""

from knowledge_index.retrieval.routers.base import (
    Complexity,
    Intent,
    QueryRouter,
    RouteDecision,
    Strategy,
)
from knowledge_index.retrieval.routers.llm_router import LLMRouter
from knowledge_index.retrieval.routers.pipeline import RouterPipeline, SupportsRetrieve

__all__ = [
    "Complexity",
    "Intent",
    "QueryRouter",
    "RouteDecision",
    "Strategy",
    "LLMRouter",
    "RouterPipeline",
    "SupportsRetrieve",
]
