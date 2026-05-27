"""Iterative / multi-hop retriever (SPEC §7.6.7).

The agentic retrieval loop. It wraps any single-shot :class:`Retriever` and, after
each round, asks a :class:`HopJudge` whether the accumulated evidence answers the
question. If not, the judge emits a focused follow-up query and the loop retrieves
again, accumulating evidence — until the judge says *done*, the hop ceiling is hit,
a follow-up repeats, or the budget runs dry.

The loop is **budget-aware** (SPEC §7.6.7): each hop reserves a fixed per-hop cost
against an optional :class:`~harness.budget.tracker.BudgetTracker`; when the next
hop cannot be afforded the loop stops early and returns what it has. This keeps a
multi-hop query from silently blowing the per-request token ceiling.

It is itself a :class:`Retriever` (returns ``list[RetrievalCandidate]``) and is
exposed as the ``strategy='iterative'`` variant (SPEC §7.6.1, line 787).
"""

from __future__ import annotations

from common.schemas import Query, RetrievalCandidate
from harness.budget.tracker import BudgetTracker
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from knowledge_index.retrieval.iterative.judge import HopJudge, LLMHopJudge
from knowledge_index.retrieval.retrievers.base import Retriever

_log = get_logger("knowledge_index.retrieval.iterative")

DEFAULT_MAX_HOPS = 3
# Conservative flat estimate of one hop's LLM judge spend; only used to gate the
# loop against a BudgetTracker. Real spend is settled at the same figure so the
# tracker's accounting stays consistent (the judge call itself is the cost).
DEFAULT_HOP_COST_USD = 0.002


class IterativeRetriever:
    """Multi-hop retrieval over an inner single-shot retriever (SPEC §7.6.7)."""

    name = "iterative"

    def __init__(
        self,
        inner: Retriever,
        judge: HopJudge | None = None,
        *,
        max_hops: int = DEFAULT_MAX_HOPS,
        hop_cost_usd: float = DEFAULT_HOP_COST_USD,
    ) -> None:
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        self._inner = inner
        self._judge = judge or LLMHopJudge()
        self._max_hops = max_hops
        self._hop_cost = hop_cost_usd

    @traced(span_name="retrieval.iterative")
    async def retrieve(
        self, query: Query, k: int, *, budget: BudgetTracker | None = None
    ) -> list[RetrievalCandidate]:
        evidence: list[RetrievalCandidate] = []
        current = query
        # Track query *text* already issued so a judge that loops back to an
        # earlier query terminates instead of spinning forever.
        seen: set[str] = {query.raw.strip().lower()}

        for hop in range(self._max_hops):
            if not self._can_afford_hop(budget):
                _log.info("iterative.budget_exhausted", hop=hop)
                break

            grant = budget.reserve(self._hop_cost) if budget is not None else None
            try:
                results = await self._inner.retrieve(current, k)
            finally:
                if budget is not None and grant is not None:
                    budget.consume(grant, self._hop_cost)
            evidence.extend(results)
            _log.info("iterative.hop", hop=hop, retrieved=len(results), total=len(evidence))

            # No point judging after the final allowed hop — we cannot act on it.
            if hop == self._max_hops - 1:
                break

            decision = await self._judge.judge(query, evidence)
            if decision.done or not decision.follow_up:
                _log.info("iterative.stop", hop=hop, reason="judge_done")
                break

            follow_up = decision.follow_up
            if follow_up.lower() in seen:
                _log.info("iterative.stop", hop=hop, reason="repeat_query")
                break
            seen.add(follow_up.lower())
            # Issue the follow-up as a fresh raw query; clear stale rewrites so the
            # inner retriever embeds the follow-up, not the original's reformulation.
            current = current.model_copy(update={"raw": follow_up, "rewrites": []})

        return _dedup_and_rank(evidence)

    def _can_afford_hop(self, budget: BudgetTracker | None) -> bool:
        return budget is None or budget.available() >= self._hop_cost


def _dedup_and_rank(evidence: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    """Collapse duplicate chunks (keep the best score), sort desc, renumber ranks.

    Evidence accrues across hops, so the same chunk can be retrieved by several
    follow-up queries. We keep the single highest-scoring occurrence per chunk and
    re-rank globally by score so the most-supported chunks surface first.
    """
    best: dict[str, RetrievalCandidate] = {}
    for cand in evidence:
        chunk_id = cand.chunk.chunk_id
        existing = best.get(chunk_id)
        if existing is None or cand.score > existing.score:
            best[chunk_id] = cand
    ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
    return [c.model_copy(update={"rank": i}) for i, c in enumerate(ranked, start=1)]


__all__ = [
    "IterativeRetriever",
    "DEFAULT_MAX_HOPS",
    "DEFAULT_HOP_COST_USD",
]
