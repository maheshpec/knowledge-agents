"""Intent-aware retrieval pipeline (SPEC §7.6.1).

``RouterPipeline`` wraps the Phase 1C :class:`~knowledge_index.retrieval.pipeline.HybridPipeline`
(any ``SupportsRetrieve``): it routes the query, selects the strategy variant,
stamps the inferred intent/filters onto the query, and executes. ``graph`` and
``iterative`` are Phase-3 stubs — they fall back to the hybrid pipeline. Because
it exposes the same ``retrieve(query, k)`` signature, the orchestrator can drop
it in wherever a ``SupportsRetrieve`` is expected.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from common.schemas import Query, RetrievalResult
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from knowledge_index.retrieval.routers.base import (
    DCI_STRATEGIES,
    QueryRouter,
    RouteDecision,
    Strategy,
)

_log = get_logger("knowledge_index.retrieval.routers.pipeline")


@runtime_checkable
class SupportsRetrieve(Protocol):
    """The retrieval surface a strategy variant must expose."""

    async def retrieve(self, query: Query, k: int) -> RetrievalResult: ...


class RouterPipeline:
    """Route → select strategy variant → execute (SPEC §7.6.1)."""

    def __init__(
        self,
        router: QueryRouter,
        hybrid: SupportsRetrieve,
        *,
        variants: dict[str, SupportsRetrieve] | None = None,
    ) -> None:
        self._router = router
        self._hybrid = hybrid
        # Per-strategy pipelines; anything unmapped (incl. graph/iterative) falls
        # back to the hybrid pipeline until Phase 3 lands the real variants.
        self._variants = variants or {}

    def _select(self, strategy: Strategy) -> SupportsRetrieve:
        variant = self._variants.get(strategy)
        if variant is not None:
            return variant
        # DCI strategies are owned by the orchestrator's dci_tool node (SPEC
        # §15.3); if one reaches RouterPipeline (e.g. caller invoked
        # ``retrieve`` directly without going through the orchestrator) we
        # degrade to hybrid rather than fail — same shape as graph/iterative.
        if strategy in ("graph", "iterative") or strategy in DCI_STRATEGIES:
            _log.info("router.fallback", strategy=strategy, to="hybrid")
        return self._variants.get("hybrid", self._hybrid)

    async def decide(self, query: Query) -> RouteDecision:
        """Expose the routing decision without executing (for inspection/eval)."""
        return await self._router.route(query)

    @traced(span_name="retrieval.routers.pipeline")
    async def retrieve(self, query: Query, k: int) -> RetrievalResult:
        decision = await self._router.route(query)
        # Propagate the inferred intent/filters so downstream stages see them.
        merged_filters = {**decision.filters, **query.filters}
        routed = query.model_copy(update={"intent": decision.intent, "filters": merged_filters})
        pipeline = self._select(decision.strategy)
        _log.info("router.execute", strategy=decision.strategy, intent=decision.intent)
        return await pipeline.retrieve(routed, k)


__all__ = ["RouterPipeline", "SupportsRetrieve"]
