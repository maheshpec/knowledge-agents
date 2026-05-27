"""Pipeline runner backed by the LangGraph orchestrator (SPEC §9.3).

:class:`OrchestratorPipelineRunner` adapts a compiled orchestrator ``app`` to the
:class:`PipelineRunner` protocol: it invokes the graph for one gold query, times
it, and reads the final state's ``candidates`` (for retrieval metrics) and
``result`` (for end-to-end + operational metrics) into a :class:`QueryOutcome`.

Kept separate from ``runner.py`` so importing :class:`EvalRunner` doesn't pull in
LangGraph; this module is imported only when actually executing the pipeline.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from common.schemas import GoldQuery
from evaluation.metrics.base import QueryOutcome


class OrchestratorPipelineRunner:
    """Drive a compiled orchestrator app for one query at a time (SPEC §9.3)."""

    def __init__(
        self,
        app: Any,
        *,
        budget_usd: float = 1.0,
        max_hops: int = 1,
        strictness: str = "strict",
    ) -> None:
        self.app = app
        self.budget_usd = budget_usd
        self.max_hops = max_hops
        self.strictness = strictness

    async def run_query(self, query: GoldQuery, *, k: int) -> QueryOutcome:
        from harness.orchestrator.graph import initial_state

        state = initial_state(
            query.query,
            budget_usd=self.budget_usd,
            k=k,
            max_hops=self.max_hops,
            strictness=self.strictness,
        )
        config = {"configurable": {"thread_id": str(uuid4())}}
        t0 = time.perf_counter()
        final = await self.app.ainvoke(state, config=config)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        candidates = final.get("candidates") or []
        result = final.get("result")
        cost = float(result.cost) if result else 0.0
        tokens_in = int(result.tokens_in) if result else 0
        tokens_out = int(result.tokens_out) if result else 0
        # "Useful" tokens: the generated answer is the product; everything fed in
        # is overhead. token_efficiency = useful / (in + out).  (SPEC §9.2)
        useful = tokens_out
        return QueryOutcome(
            gold=query,
            candidates=candidates,
            generation=result,
            latency_ms=latency_ms,
            cost_usd=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            useful_tokens=useful,
        )


__all__ = ["OrchestratorPipelineRunner"]
