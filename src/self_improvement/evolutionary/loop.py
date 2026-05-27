"""The AlphaEvolve-style evolutionary loop (SPEC §8.2).

Each generation: mutate + cross the current population, evaluate offspring on the
held-out eval set, run the adversarial reviewer, then select the top-k by composite
score into the next generation. The loop stops when the requested generation count
is reached or the budget guard trips.

The loop never touches the frozen test set (SPEC §9.1) — qualification uses only
the dev/rotating metrics the evaluator attaches; the frozen-set check is a separate
pre-PR step (§8.4) outside this class. Every collaborator (evaluator, reviewer,
budget, optional ledger) is injected behind a protocol so real implementations and
test stubs are interchangeable.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from self_improvement.evolutionary.genome import crossover, mutate, random_config
from self_improvement.evolutionary.selection import (
    DEFAULT_DELTA_THRESHOLD,
    ScorePolicy,
    composite_score,
    qualifies,
    select,
)
from self_improvement.evolutionary.types import (
    BudgetGuard,
    Candidate,
    Evaluator,
    EvolutionReport,
    ExperimentLedger,
    MutationRecord,
    Reviewer,
)
from self_improvement.registry.pipeline_config import PipelineConfig
from self_improvement.registry.registry import ComponentRegistry


class EvolutionaryLoop:
    """Evolve pipeline configs toward a better composite score (SPEC §8.2)."""

    def __init__(
        self,
        registry: ComponentRegistry,
        evaluator: Evaluator,
        reviewer: Reviewer,
        budget: BudgetGuard,
        *,
        rng: random.Random | None = None,
        score_policy: ScorePolicy | None = None,
        delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
        ledger: ExperimentLedger | None = None,
        seed_config: PipelineConfig | None = None,
    ) -> None:
        self.registry = registry
        self.evaluator = evaluator
        self.reviewer = reviewer
        self.budget = budget
        self.rng = rng or random.Random()
        self.policy = score_policy or ScorePolicy()
        self.delta_threshold = delta_threshold
        self.ledger = ledger
        # A known-good config anchors generation 0 so the run always contains the
        # current production pipeline as a comparison point.
        self.seed_config = seed_config or PipelineConfig()
        self.run_id = uuid.uuid4().hex

    # --- candidate construction ---

    def _new_candidate(
        self,
        config: PipelineConfig,
        *,
        generation: int,
        parent_ids: list[str],
        mutation: MutationRecord,
    ) -> Candidate:
        return Candidate(
            candidate_id=uuid.uuid4().hex,
            config=config,
            generation=generation,
            parent_ids=parent_ids,
            mutation=mutation,
        )

    def seed_population(self, size: int) -> list[Candidate]:
        """Generation 0: the anchor config + random draws from the registry."""
        seed = self._new_candidate(
            self.seed_config,
            generation=0,
            parent_ids=[],
            mutation=MutationRecord(type="seed", component="anchor"),
        )
        population = [seed]
        while len(population) < max(1, size):
            cfg = random_config(self.registry, self.rng)
            population.append(
                self._new_candidate(
                    cfg,
                    generation=0,
                    parent_ids=[],
                    mutation=MutationRecord(type="seed"),
                )
            )
        return population

    def mutate_and_cross(self, population: list[Candidate], generation: int) -> list[Candidate]:
        """Produce one offspring per parent: mutate, or cross two parents (SPEC §8.2)."""
        offspring: list[Candidate] = []
        for parent in population:
            if len(population) >= 2 and self.rng.random() < 0.5:
                mate = self.rng.choice([c for c in population if c is not parent])
                child_cfg, record = crossover(parent.config, mate.config, self.rng)
                parents = [parent.candidate_id, mate.candidate_id]
            else:
                child_cfg, record = mutate(parent.config, self.registry, self.rng)
                parents = [parent.candidate_id]
            offspring.append(
                self._new_candidate(
                    child_cfg, generation=generation, parent_ids=parents, mutation=record
                )
            )
        return offspring

    def select(self, candidates: list[Candidate], k: int) -> list[Candidate]:
        """Top-``k`` by composite score, dropping reviewer-rejected candidates."""
        return select(candidates, k, self.policy)

    # --- evaluation pipeline for one batch ---

    async def _assess(self, candidates: list[Candidate]) -> list[Candidate]:
        """Evaluate → review → score one batch, charging the budget for its cost."""
        evaluated = await self.evaluator.evaluate_batch(candidates)
        reviewed = await self.reviewer.review_batch(evaluated)
        for c in reviewed:
            c.score = composite_score(c, self.policy)
            self.budget.charge(cost_usd=c.cost_usd, compute_seconds=c.compute_seconds)
            if self.ledger is not None:
                await self.ledger.append(c)
        return reviewed

    # --- the loop ---

    async def run(self, generations: int, population_size: int) -> EvolutionReport:
        """Evolve for up to ``generations`` rounds (SPEC §8.2)."""
        population = await self._assess(self.seed_population(population_size))
        baseline = max((c.score or 0.0 for c in population), default=0.0)

        history: list[dict[str, Any]] = [self._gen_summary(0, population)]
        stopped = "completed"
        gens_run = 0

        for gen in range(1, generations + 1):
            if self.budget.exhausted():
                stopped = "budget_exhausted"
                break
            offspring = await self._assess(self.mutate_and_cross(population, generation=gen))
            population = self.select(population + offspring, population_size)
            self.budget.tick_generation()
            gens_run = gen
            history.append(self._gen_summary(gen, population))
            if self.budget.exhausted():
                stopped = "budget_exhausted"
                break

        best_overall = max(
            population,
            key=lambda c: c.score if c.score is not None else float("-inf"),
            default=None,
        )
        qualified = [
            c
            for c in population
            if qualifies(
                c,
                baseline_score=baseline,
                policy=self.policy,
                delta_threshold=self.delta_threshold,
            )
        ]
        best = max(qualified, key=lambda c: c.score or 0.0, default=None)

        return EvolutionReport(
            run_id=self.run_id,
            generations_run=gens_run,
            stopped_reason=stopped,
            baseline_score=baseline,
            best=best,
            best_overall=best_overall,
            population=population,
            history=history,
        )

    def _gen_summary(self, generation: int, population: list[Candidate]) -> dict[str, Any]:
        scores = [c.score for c in population if c.score is not None]
        return {
            "generation": generation,
            "n": len(population),
            "best_score": max(scores, default=0.0),
            "mean_score": (sum(scores) / len(scores)) if scores else 0.0,
        }


__all__ = ["EvolutionaryLoop"]
