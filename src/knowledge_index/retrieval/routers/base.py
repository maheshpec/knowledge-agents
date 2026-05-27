"""Query routing contract + decision schema (SPEC §7.6.1).

A :class:`QueryRouter` inspects an incoming :class:`~common.schemas.Query` and
picks a retrieval *strategy* (which pipeline variant to run) plus the query's
*intent* and *expected complexity*. The orchestrator/pipeline then executes the
chosen variant. ``RouteDecision`` is the structured output; it lives here rather
than in ``common/schemas`` because routing is a retrieval-layer concern.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from common.schemas import Query

# Retrieval strategy variants (SPEC §7.6.1). 'graph' and 'iterative' are stubbed
# for Phase 3 and fall back to 'hybrid' in the RouterPipeline.
Strategy = Literal["naive", "hybrid", "graph", "iterative"]
Intent = Literal["lookup", "synthesis", "comparison", "relational"]
Complexity = Literal["low", "med", "high"]


class RouteDecision(BaseModel):
    """The router's structured verdict for a query (SPEC §7.6.1)."""

    strategy: Strategy = "hybrid"
    intent: Intent = "lookup"
    expected_complexity: Complexity = "low"
    filters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


@runtime_checkable
class QueryRouter(Protocol):
    """Decide how to retrieve for a given query (SPEC §7.6.1)."""

    name: str

    async def route(self, query: Query) -> RouteDecision: ...


__all__ = [
    "Strategy",
    "Intent",
    "Complexity",
    "RouteDecision",
    "QueryRouter",
]
