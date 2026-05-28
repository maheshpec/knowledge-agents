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

# Retrieval strategy variants (SPEC §7.6.1, §15.2). 'graph' and 'iterative' are
# stubbed for Phase 3 and fall back to 'hybrid' in the RouterPipeline. The DCI
# strategies ('dci', 'dci_then_vector', 'vector_then_dci') are Phase 5: the
# orchestrator runs them through a dedicated ``dci_tool`` node so they bypass
# the vector RouterPipeline entirely (which falls them back to 'hybrid' if it
# ever sees them — see RouterPipeline._select).
Strategy = Literal[
    "naive",
    "hybrid",
    "graph",
    "iterative",
    "dci",
    "dci_then_vector",
    "vector_then_dci",
]
Intent = Literal["lookup", "synthesis", "comparison", "relational"]
Complexity = Literal["low", "med", "high"]

# Strategy values that demand the orchestrator's DCI node at least once. The
# router-pipeline reads this to know it must fall back to vector hybrid when a
# DCI strategy reaches it (the orchestrator should have intercepted first).
DCI_STRATEGIES: frozenset[str] = frozenset({"dci", "dci_then_vector", "vector_then_dci"})


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
    "DCI_STRATEGIES",
]
