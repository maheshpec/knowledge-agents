"""Wire the Phase 4 modules into one runnable self-improvement loop (SPEC §8).

The evolutionary loop (§8.2) was deliberately written against narrow protocols
(``Evaluator``, ``Reviewer``, ``BudgetGuard``, ``ExperimentLedger``) so the
heavier Phase 4O ledger / budget guard and the §8.4 PR generator can drop in
unchanged. This module supplies the small adapters that close the gap:

- :class:`LedgerAdapter` — accepts a :class:`Candidate` from the loop and
  materialises an :class:`~self_improvement.ledger.Experiment` into the §8.2.1
  JSONL ledger.
- :class:`BudgetGuardAdapter` — exposes the §8.5 :class:`BudgetGuard`'s hard
  kill switch via the loop's lightweight ``BudgetGuard`` protocol, so a tripped
  ceiling drains :meth:`exhausted` instead of crashing mid-generation.
- :func:`build_loop` — the one-call factory the CLI and the acceptance test
  use to compose registry + loop + ledger + budget guard.

The factory is the integration point: any caller that hands in an
``Evaluator`` and a ``Reviewer`` (and, for PR creation, a :class:`PRGenerator`)
gets a runnable end-to-end loop with frozen-set isolation, the real kill switch,
the durable ledger, and a no-merge PR surface — exactly the Phase 4 contract.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from self_improvement.budget_guard import BudgetConfig, BudgetExceeded, BudgetGuard
from self_improvement.evolutionary import (
    BudgetGuard as EvoBudgetGuard,
)
from self_improvement.evolutionary import (
    Candidate,
    EvolutionaryLoop,
    Evaluator,
    Reviewer,
)
from self_improvement.evolutionary.types import MutationRecord as EvoMutationRecord
from self_improvement.ledger import Experiment
from self_improvement.ledger import MutationRecord as LedgerMutationRecord
from self_improvement.ledger.ledger import JSONLLedger
from self_improvement.registry import ComponentRegistry
from self_improvement.registry.pipeline_config import PipelineConfig


# --- ledger ------------------------------------------------------------------


class LedgerAdapter:
    """Maps loop :class:`Candidate`\\s onto §8.2.1 :class:`Experiment` records.

    Implements the loop's :class:`ExperimentLedger` protocol (single ``append``)
    by translating each candidate into a full ``Experiment`` and delegating to
    the underlying :class:`JSONLLedger`. The richer ``status`` / ``cost_usd`` /
    ``eval_results`` fields the ledger schema carries are populated from what
    the candidate has accrued by the time the loop appends it.
    """

    def __init__(self, ledger: JSONLLedger, run_id: str) -> None:
        self._ledger = ledger
        self._run_id = run_id

    async def append(self, record: Any) -> None:
        if not isinstance(record, Candidate):
            # Allow passing a pre-built Experiment (the script / test may bypass
            # the translation, e.g. when seeding lineage explicitly).
            await self._ledger.append(record)
            return
        exp = _candidate_to_experiment(record, self._run_id)
        await self._ledger.append(exp)


def _candidate_to_experiment(c: Candidate, run_id: str) -> Experiment:
    """Translate a :class:`Candidate` into a ledger :class:`Experiment` record."""
    mutation = _translate_mutation(c.mutation)
    # Reviewer-rejected candidates land as ``rejected``; cleared ones as
    # ``accepted`` (the verdict gates PR creation). Unreviewed→evaluated.
    if c.verdict == "accept":
        status = "accepted"
    elif c.verdict == "reject":
        status = "rejected"
    elif c.verdict is not None:
        status = "reviewed"
    else:
        status = "evaluated"
    return Experiment(
        experiment_id=c.candidate_id,
        parent_ids=list(c.parent_ids),
        generation=c.generation,
        run_id=run_id,
        config=c.config,
        mutation=mutation,
        status=status,
        cost_usd=c.cost_usd,
        compute_seconds=c.compute_seconds,
    )


def _translate_mutation(m: EvoMutationRecord | None) -> LedgerMutationRecord | None:
    if m is None:
        return None
    # The two records share the same shape; restate it on the ledger side so
    # the JSONL line validates against the §8.2.1 schema.
    return LedgerMutationRecord(type=m.type, component=m.component or "", change=dict(m.change))


# --- budget guard ------------------------------------------------------------


class BudgetGuardAdapter:
    """Expose the §8.5 :class:`BudgetGuard` via the loop's evolutionary protocol.

    The §8.5 guard raises :class:`BudgetExceeded` the instant a ceiling is
    breached; the loop's protocol communicates exhaustion via the ``exhausted``
    poll. We catch the exception so a tripped ceiling drains the loop on its
    next poll instead of aborting mid-generation, and surface ``inner.tripped``
    via ``exhausted`` — the kill switch is therefore observed, one-way, and
    never silently bypassed (a re-poll always re-trips).
    """

    def __init__(self, inner: BudgetGuard, run_id: str) -> None:
        self._inner = inner
        self._run_id = run_id
        # Tracks the latest started generation so per-gen compute is attributed.
        self._current_gen = 0

    @property
    def inner(self) -> BudgetGuard:
        return self._inner

    def charge(self, *, cost_usd: float = 0.0, compute_seconds: float = 0.0) -> None:
        try:
            if cost_usd:
                self._inner.record_spend(self._run_id, cost_usd)
            if compute_seconds:
                self._inner.record_compute(self._run_id, self._current_gen, compute_seconds)
        except BudgetExceeded:
            # Tripping is the signal; the loop polls ``exhausted`` and stops.
            return

    def tick_generation(self) -> None:
        try:
            self._current_gen = self._inner.start_generation(self._run_id)
        except BudgetExceeded:
            return

    def exhausted(self) -> bool:
        return self._inner.tripped


# --- factory -----------------------------------------------------------------


# The CLI / acceptance test inject these by name; default to the lightweight
# protocol type so callers can pass any compatible implementation.
EvaluatorFactory = Callable[[ComponentRegistry], Evaluator]
ReviewerFactory = Callable[[], Reviewer]


def build_loop(
    *,
    evaluator: Evaluator,
    reviewer: Reviewer,
    registry: ComponentRegistry | None = None,
    ledger_root: str | Path = "experiments",
    run_id: str | None = None,
    budget_config: BudgetConfig | None = None,
    seed_config: PipelineConfig | None = None,
    delta_threshold: float | None = None,
    rng: random.Random | None = None,
) -> tuple[EvolutionaryLoop, JSONLLedger, BudgetGuard]:
    """Compose a self-improvement loop wired to the §8.2.1 ledger + §8.5 guard.

    Returns the loop along with the underlying ledger and budget guard so the
    caller can introspect them after the run (acceptance test asserts the kill
    switch tripped, the CLI surfaces the storage path, etc.).
    """
    registry = registry or ComponentRegistry.from_yaml()
    inner_budget = BudgetGuard(budget_config or BudgetConfig())
    ledger = JSONLLedger(ledger_root)
    rid = run_id or _new_run_id()
    budget_adapter = BudgetGuardAdapter(inner_budget, rid)
    ledger_adapter = LedgerAdapter(ledger, rid)
    extras: dict[str, Any] = {}
    if delta_threshold is not None:
        extras["delta_threshold"] = delta_threshold
    if seed_config is not None:
        extras["seed_config"] = seed_config
    if rng is not None:
        extras["rng"] = rng
    loop = EvolutionaryLoop(
        registry=registry,
        evaluator=evaluator,
        reviewer=reviewer,
        budget=budget_adapter,
        ledger=ledger_adapter,
        **extras,
    )
    # Stamp the loop's run id so ledger records and the manifest match up.
    loop.run_id = rid
    return loop, ledger, inner_budget


def _new_run_id() -> str:
    import uuid

    return uuid.uuid4().hex


# Re-export so the script and tests have one import surface.
__all__ = [
    "BudgetGuardAdapter",
    "LedgerAdapter",
    "build_loop",
    "EvaluatorFactory",
    "ReviewerFactory",
    "EvoBudgetGuard",
]
