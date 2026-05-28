"""Phase 4 acceptance test (SPEC §10 Phase 4 / §11): end-to-end self-improvement.

Drives the wired loop (registry + evolutionary + ledger + reviewer + pr_gen +
budget guard) on a small offline population and asserts the SPEC contract:

(a) the loop surfaces ≥1 candidate that improves the primary metric beyond the
    delta threshold (SPEC §8.2 qualification);
(b) that improvement still holds on the rotating eval set (Goodhart guard,
    SPEC §8.2);
(c) the adversarial reviewer (§8.3) clears the candidate;
(d) the PR generator (§8.4) emits a complete evidence package and opens a
    (mocked) GitHub PR with **no merge** surface (anti-pattern §13).

Plus two structural guards Phase 4 requires:

- the frozen test set is *unreachable* during evolution mode (SPEC §9.1 / §13);
- the §8.5 budget kill switch actually stops the loop when tripped.

LLM and GitHub clients are mocked end-to-end so the test is offline and
deterministic; no API keys, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from common.errors import FrozenSetIsolationError
from evaluation.datasets.loader import evolution_mode, load_dataset
from evaluation.metrics.base import MetricResult
from evaluation.runners.runner import EvalReport, QueryReport
from self_improvement.budget_guard import BudgetConfig
from self_improvement.evolutionary import Candidate, ScorePolicy
from self_improvement.integration import build_loop
from self_improvement.ledger import RunManifest
from self_improvement.pr_gen import (
    AcceptedCandidate,
    LineageEntry,
    PRGenerator,
    RecordingBranchWriter,
    RecordingGitHubClient,
    ReviewerReport,
    build_evidence_package,
)
from self_improvement.registry import ComponentRegistry
from self_improvement.reviewer import AdversarialReviewer, ReviewThresholds

PRIMARY = "recall@5"


# --- synthetic evaluator -----------------------------------------------------
#
# Designed so:
#  * one mutation gene ("reranker_top_k" delta) lifts recall@5 by ~0.10,
#  * that gain survives on a separate rotating signal (Goodhart guard),
#  * per-query reports support the reviewer's leakage / narrow-slice / regression
#    checks; the seed population spans a small enough range that a +0.10 candidate
#    clears the noise band.


class TaggedEvaluator:
    """Returns deterministic, lineage-friendly metrics keyed off the genome.

    The *primary signal* is ``reranker_top_k``: an offset of ``+N`` from the seed
    lifts ``recall@5`` by ``0.01 * N`` on both dev and rotating sets. Seed
    configs sample within a tight band so the noise band is non-zero but small,
    and a +0.10 candidate exceeds it (2 sigmas of seed std).
    """

    name = "tagged"

    def __init__(self) -> None:
        # candidate_id → (dev_report, rotating_report)
        self.reports: dict[str, tuple[EvalReport, EvalReport]] = {}
        # ids of the gen-0 seed batch; the noise-band check pulls its spread.
        self.seed_ids: set[str] = set()

    @staticmethod
    def _score(c: Candidate) -> float:
        # Seed configs (anchor + random) cluster near 0.60–0.62; "improver"
        # tag in chunker_params or larger reranker_top_k pulls the score up.
        base = 0.60 + 0.001 * (c.config.reranker_top_k - 10)
        if c.config.chunker_params.get("improver"):
            return min(0.95, base + 0.10)
        return base

    def _eval_report(self, dataset: str, primary: float) -> EvalReport:
        # 20 per-query rows; spread the primary metric across queries so the
        # reviewer's "broad improvement" check (≥30% improved) can succeed.
        per_query = [
            QueryReport(
                query_id=f"{dataset}-q{i:02d}",
                metrics={PRIMARY: primary + (i % 5 - 2) * 0.01},
                latency_ms=100.0,
                cost_usd=0.001,
            )
            for i in range(20)
        ]
        return EvalReport(
            dataset=dataset, n=len(per_query), aggregated={PRIMARY: primary}, per_query=per_query
        )

    async def evaluate_batch(self, candidates: list[Candidate]) -> list[Candidate]:
        for c in candidates:
            dev = self._score(c)
            rot = dev - 0.005  # rotating tracks dev (Goodhart guard satisfiable)
            c.metrics = {PRIMARY: dev}
            c.rotating_metrics = {PRIMARY: rot}
            c.cost_usd = 0.001
            c.compute_seconds = 0.05
            self.reports[c.candidate_id] = (
                self._eval_report("dev", dev),
                self._eval_report("rotating", rot),
            )
            # The first batch the loop hands in is the seed population (gen 0).
            if c.generation == 0:
                self.seed_ids.add(c.candidate_id)
        return candidates


class ImproverInjectingReviewer:
    """Loop-side reviewer: clears every candidate; injects one "improver" by hand.

    The loop's :class:`Reviewer` protocol only sets ``verdict``; the *actual*
    adversarial review (SPEC §8.3) happens in the (c) assertion below using
    ``AdversarialReviewer`` against the recorded reports. This stub also flips
    one offspring's ``chunker_params['improver']=True`` so the evaluator has a
    candidate to lift on the next generation (deterministic test signal).
    """

    def __init__(self, evaluator: TaggedEvaluator) -> None:
        self._evaluator = evaluator
        self._injected = False

    async def review_batch(self, candidates: list[Candidate]) -> list[Candidate]:
        for c in candidates:
            # Inject the improver exactly once, on the first non-seed candidate
            # we see, then re-score it so its candidate metrics reflect the gain.
            if not self._injected and c.generation > 0:
                c.config.chunker_params["improver"] = True
                # Re-evaluate this single candidate so its metrics + reports
                # match the post-inject config.
                await self._evaluator.evaluate_batch([c])
                self._injected = True
            c.verdict = "accept"
        return candidates


# --- the acceptance test -----------------------------------------------------


@pytest.mark.asyncio
async def test_phase4_acceptance(tmp_path):
    registry = ComponentRegistry.from_yaml()
    evaluator = TaggedEvaluator()
    reviewer_stub = ImproverInjectingReviewer(evaluator)

    loop, ledger, budget = build_loop(
        evaluator=evaluator,
        reviewer=reviewer_stub,
        registry=registry,
        ledger_root=tmp_path / "experiments",
        run_id="run-acceptance",
        budget_config=BudgetConfig(
            max_generations=5,
            max_compute_hours_per_gen=2.0,
            max_usd_per_run=10.0,
            daily_usd_ceiling=100.0,
        ),
        delta_threshold=0.005,
        score_policy=ScorePolicy(primary_metric=PRIMARY),
    )
    ledger.write_manifest(
        RunManifest(
            run_id=loop.run_id,
            generations=3,
            population_size=4,
            dataset_refs=["dev", "rotating"],
            budget=budget.config.model_dump(),
        )
    )

    # Frozen-set isolation must hold for the entire search (SPEC §13).
    with evolution_mode():
        with pytest.raises(FrozenSetIsolationError):
            load_dataset("frozen")
        report = await loop.run(generations=3, population_size=4)

    # ----- (a) at least one candidate beats baseline by > delta on dev -------
    assert report.best is not None, "no qualified candidate; loop should surface one"
    delta_dev = report.best.metrics[PRIMARY] - report.baseline_score
    assert delta_dev > 0.005, f"dev gain {delta_dev:.4f} did not clear delta threshold"

    # ----- (b) the gain holds on the rotating set (Goodhart guard) -----------
    # Rotating baseline = best seed's rotating score; any seed-generation report
    # works because TaggedEvaluator's rotating tracks dev minus a small constant.
    baseline_rot = max(evaluator.reports[sid][1].aggregated[PRIMARY] for sid in evaluator.seed_ids)
    assert report.best.rotating_metrics is not None
    delta_rot = report.best.rotating_metrics[PRIMARY] - baseline_rot
    assert delta_rot > 0.005, (
        f"dev gain {delta_dev:.4f} did NOT survive on rotating set (Δ={delta_rot:.4f}); "
        "Goodhart guard would reject this PR"
    )

    # ----- (c) the adversarial reviewer clears the candidate -----------------
    candidate_dev, _ = evaluator.reports[report.best.candidate_id]
    seed_dev_reports = [evaluator.reports[sid][0] for sid in evaluator.seed_ids]
    baseline_dev_report = max(seed_dev_reports, key=lambda r: r.aggregated[PRIMARY])
    seed_scores = [r.aggregated[PRIMARY] for r in seed_dev_reports]

    adversarial = AdversarialReviewer(
        completion_fn=None,  # deterministic checks only — no LLM call needed
        thresholds=ReviewThresholds(primary_metric=PRIMARY, min_improved_fraction=0.10),
    )
    verdict = await adversarial.review(
        candidate_dev,
        baseline_dev_report,
        seed_scores=seed_scores,
        train_query_ids=set(),  # no leakage from training/seed query ids
    )
    assert verdict.verdict == "accept", (
        f"adversarial reviewer did not clear best candidate: "
        f"verdict={verdict.verdict} failed={[c.name for c in verdict.failed_checks]} "
        f"critique={verdict.critique}"
    )

    # ----- (d) pr_gen produces evidence + opens a (mocked) PR, no merge ------
    github = RecordingGitHubClient(repo="acme/knowledge-agent")
    writer = RecordingBranchWriter()
    pr_gen = PRGenerator(github, writer, base_branch="main")
    accepted = AcceptedCandidate(
        experiment_id=report.best.candidate_id,
        run_id=loop.run_id,
        config=report.best.config,
        baseline_config=loop.seed_config,
        before={PRIMARY: MetricResult(name=PRIMARY, value=report.baseline_score)},
        after={PRIMARY: MetricResult(name=PRIMARY, value=report.best.metrics[PRIMARY])},
        lineage=[
            LineageEntry(
                experiment_id=c.candidate_id,
                generation=c.generation,
                mutation_summary=(c.mutation.type if c.mutation else ""),
                config_hash=c.config_hash,
            )
            for c in report.population
        ],
        reviewer=ReviewerReport(
            verdict="accept",
            critique=verdict.critique,
            checks={c.name: c.passed for c in verdict.checks},
        ),
    )
    # Current config text is fed in by the caller (SPEC §8.4 keeps pr_gen pure).
    current_yaml = (loop.registry and "retrieval: {}") or ""
    generated = await pr_gen.generate(accepted, current_config_text=current_yaml)

    # Evidence package is complete (all SPEC §8.4 fields populated).
    evidence = build_evidence_package(accepted)
    assert evidence.experiment_id == report.best.candidate_id
    assert evidence.metric_deltas, "no metric deltas in evidence package"
    assert any(d.name == PRIMARY and d.after > d.before for d in evidence.metric_deltas)
    assert evidence.reviewer.verdict == "accept"
    assert evidence.lineage, "lineage not included in evidence package"

    # PR opened as a draft, labelled for review, on a self-improve branch.
    assert len(github.opened) == 1
    opened = github.opened[0]
    assert opened.draft is True
    assert "self-improvement" in opened.labels
    assert generated.branch.startswith("self-improve/")
    assert writer.branch is not None and writer.branch[1] == "main"

    # No-auto-merge is structural: the protocol exposes no merge method.
    assert not hasattr(github, "merge_pull_request")
    assert not hasattr(github, "merge")

    # Ledger persisted every appended candidate (one line each generation).
    appended = await ledger.query(lambda e: e.run_id == loop.run_id)
    assert len(appended) >= 4, f"ledger only persisted {len(appended)} experiments"
    assert any(e.experiment_id == report.best.candidate_id for e in appended)


# --- structural guards Phase 4 also requires --------------------------------


@pytest.mark.asyncio
async def test_budget_kill_switch_stops_the_loop(tmp_path):
    """A breached §8.5 ceiling drains :meth:`exhausted` and stops generation."""

    class _CheapEvaluator:
        async def evaluate_batch(self, candidates: list[Candidate]) -> list[Candidate]:
            for c in candidates:
                c.metrics = {PRIMARY: 0.6}
                c.rotating_metrics = {PRIMARY: 0.6}
                c.cost_usd = 5.0  # large enough to trip the per-run ceiling fast
                c.compute_seconds = 0.0
            return candidates

    class _NoopReviewer:
        async def review_batch(self, candidates: list[Candidate]) -> list[Candidate]:
            for c in candidates:
                c.verdict = "accept"
            return candidates

    loop, _, budget = build_loop(
        evaluator=_CheapEvaluator(),
        reviewer=_NoopReviewer(),
        ledger_root=tmp_path / "experiments",
        run_id="run-budget",
        budget_config=BudgetConfig(
            max_generations=20,
            max_compute_hours_per_gen=4.0,
            max_usd_per_run=10.0,  # 2 charges of $5 trip this
            daily_usd_ceiling=1000.0,
        ),
    )
    with evolution_mode():
        report = await loop.run(generations=20, population_size=4)

    assert budget.tripped is True
    assert budget.trip_reason and "max_usd_per_run" in budget.trip_reason
    assert report.stopped_reason == "budget_exhausted"
    assert report.generations_run < 20, "loop ignored the kill switch"


def test_no_merge_method_on_github_client():
    """SPEC §8.4 / anti-pattern §13: structural no-auto-merge guarantee."""
    from self_improvement.pr_gen.github import GitHubClient, RecordingGitHubClient

    # Protocol surface is just open_pull_request (declared by the SPEC).
    declared = {m for m in dir(GitHubClient) if not m.startswith("_")}
    assert declared == {"open_pull_request"}, declared
    # The in-memory test double matches the protocol — no merge attribute.
    client: Any = RecordingGitHubClient()
    for name in ("merge", "merge_pull_request", "squash_merge", "rebase_and_merge"):
        assert not hasattr(client, name), f"GitHubClient unexpectedly exposes '{name}'"
