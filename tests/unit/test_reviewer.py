"""Tests for the adversarial reviewer (SPEC §8.3).

Covers each deterministic check in isolation, verdict gating (accept / reject /
needs_more_evidence and the ``gates_pr`` gate), the adversarial LLM pass (which
may only tighten the verdict), and lineage-driven review against a fake §8.2.1
ledger. Everything is offline: ``completion_fn`` is injected.
"""

from __future__ import annotations

import statistics

import pytest

from evaluation.runners.runner import EvalReport, QueryReport
from self_improvement.reviewer import (
    AdversarialReviewer,
    LineageEntry,
    ReviewThresholds,
    derive_baseline_and_seeds,
)

METRIC = "recall@5"


def _report(
    recalls: dict[str, float],
    *,
    latency: float = 10.0,
    cost: float = 0.001,
    dataset: str = "frozen",
) -> EvalReport:
    qrs = [
        QueryReport(query_id=qid, metrics={METRIC: r}, latency_ms=latency, cost_usd=cost)
        for qid, r in recalls.items()
    ]
    agg = statistics.fmean(recalls.values()) if recalls else 0.0
    return EvalReport(dataset=dataset, n=len(qrs), aggregated={METRIC: agg}, per_query=qrs)


@pytest.fixture
def reviewer() -> AdversarialReviewer:
    return AdversarialReviewer()  # no LLM: pure deterministic checks


# --------------------------------------------------------------------------- #
# Leakage check
# --------------------------------------------------------------------------- #


def test_leakage_clean(reviewer):
    cand = _report({"q1": 1.0, "q2": 1.0})
    res = reviewer.check_leakage(cand, train_query_ids={"t1", "t2"})
    assert res.passed and res.critical


def test_leakage_detects_overlap(reviewer):
    cand = _report({"q1": 1.0, "q2": 1.0})
    res = reviewer.check_leakage(cand, train_query_ids={"q2", "t1"})
    assert not res.passed
    assert res.critical
    assert res.detail["overlap"] == ["q2"]


# --------------------------------------------------------------------------- #
# Narrow-slice check
# --------------------------------------------------------------------------- #


def test_narrow_slice_broad_gain_passes(reviewer):
    baseline = _report({"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0})
    cand = _report({"q1": 1.0, "q2": 1.0, "q3": 0.0, "q4": 0.0})  # 2/4 improved
    res = reviewer.check_narrow_slice(cand, baseline)
    assert res.passed
    assert res.detail["improved"] == 2


def test_narrow_slice_confined_gain_fails_noncritical(reviewer):
    baseline = _report({"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0})
    cand = _report({"q1": 1.0, "q2": 0.0, "q3": 0.0, "q4": 0.0})  # 1/4 improved
    res = reviewer.check_narrow_slice(cand, baseline)
    assert not res.passed
    assert not res.critical  # demotes to needs_more_evidence, not reject


def test_narrow_slice_no_baseline_is_noncritical_fail(reviewer):
    res = reviewer.check_narrow_slice(_report({"q1": 1.0}), None)
    assert not res.passed and not res.critical


# --------------------------------------------------------------------------- #
# Noise-band check
# --------------------------------------------------------------------------- #


def test_noise_band_real_gain_passes(reviewer):
    baseline = _report({"q1": 0.5, "q2": 0.5})  # agg 0.5
    cand = _report({"q1": 0.8, "q2": 0.8})  # agg 0.8 -> +0.30
    res = reviewer.check_noise_band(cand, baseline, seed_scores=[0.48, 0.50, 0.52])
    assert res.passed and res.critical
    assert res.detail["improvement"] == pytest.approx(0.30)


def test_noise_band_within_variance_fails_critical(reviewer):
    baseline = _report({"q1": 0.5, "q2": 0.5})
    cand = _report({"q1": 0.52, "q2": 0.52})  # +0.02, inside 2σ of seed spread
    res = reviewer.check_noise_band(cand, baseline, seed_scores=[0.48, 0.50, 0.52])
    assert not res.passed
    assert res.critical


def test_noise_band_insufficient_seeds_is_noncritical(reviewer):
    baseline = _report({"q1": 0.5})
    cand = _report({"q1": 0.9})
    res = reviewer.check_noise_band(cand, baseline, seed_scores=[0.5])
    assert not res.passed and not res.critical


# --------------------------------------------------------------------------- #
# Regression check
# --------------------------------------------------------------------------- #


def test_regression_within_thresholds_passes(reviewer):
    baseline = _report({"q1": 0.5}, latency=10.0, cost=0.001)
    cand = _report({"q1": 0.5}, latency=11.0, cost=0.001)  # +10% latency
    res = reviewer.check_regression(cand, baseline)
    assert res.passed and res.critical


def test_regression_latency_blowout_fails_critical(reviewer):
    baseline = _report({"q1": 0.5}, latency=10.0)
    cand = _report({"q1": 0.5}, latency=13.0)  # +30% > 20%
    res = reviewer.check_regression(cand, baseline)
    assert not res.passed and res.critical
    assert res.detail["latency_pct"] == pytest.approx(0.30)


def test_regression_cost_blowout_fails_critical(reviewer):
    baseline = _report({"q1": 0.5}, cost=0.0010)
    cand = _report({"q1": 0.5}, cost=0.0013)  # +30% > 20%
    res = reviewer.check_regression(cand, baseline)
    assert not res.passed and res.critical


# --------------------------------------------------------------------------- #
# Verdict gating
# --------------------------------------------------------------------------- #


def _clean_case() -> tuple[EvalReport, EvalReport, list[float]]:
    """A candidate that passes every check, plus its baseline and seed scores."""
    baseline = _report({"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 0.5}, latency=10.0, cost=0.001)
    cand = _report({"q1": 0.9, "q2": 0.9, "q3": 0.8, "q4": 0.8}, latency=10.0, cost=0.001)
    seeds = [0.48, 0.50, 0.52]
    return cand, baseline, seeds


@pytest.mark.asyncio
async def test_verdict_accept_when_all_pass(reviewer):
    cand, baseline, seeds = _clean_case()
    v = await reviewer.review(cand, baseline, seed_scores=seeds, train_query_ids={"t1"})
    assert v.verdict == "accept"
    assert v.gates_pr is True
    assert all(c.passed for c in v.checks)


@pytest.mark.asyncio
async def test_verdict_reject_on_critical_failure(reviewer):
    cand, baseline, seeds = _clean_case()
    # Leak an eval query into the training set -> critical failure.
    v = await reviewer.review(cand, baseline, seed_scores=seeds, train_query_ids={"q1"})
    assert v.verdict == "reject"
    assert v.gates_pr is False


@pytest.mark.asyncio
async def test_verdict_needs_more_evidence_on_noncritical_failure(reviewer):
    # Narrow gain (1/4) but no leakage, real (large) gain, no regression.
    baseline = _report({"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 0.5})
    cand = _report({"q1": 1.0, "q2": 0.5, "q3": 0.5, "q4": 0.5})  # only q1 improves
    v = await reviewer.review(
        cand, baseline, seed_scores=[0.49, 0.50, 0.51], train_query_ids={"t1"}
    )
    assert v.verdict == "needs_more_evidence"
    assert v.gates_pr is False


# --------------------------------------------------------------------------- #
# Adversarial LLM pass
# --------------------------------------------------------------------------- #


def _fake_completion(payload: str):
    async def _complete(_prompt: str) -> str:
        return payload

    return _complete


@pytest.mark.asyncio
async def test_llm_can_downgrade_accept_to_needs_more_evidence():
    cand, baseline, seeds = _clean_case()
    llm = _fake_completion(
        '{"verdict": "needs_more_evidence", "concerns": ["small sample"], "critique": "thin"}'
    )
    reviewer = AdversarialReviewer(completion_fn=llm)
    v = await reviewer.review(cand, baseline, seed_scores=seeds, train_query_ids={"t1"})
    assert v.verdict == "needs_more_evidence"
    assert "small sample" in v.concerns
    assert v.critique == "thin"


@pytest.mark.asyncio
async def test_llm_cannot_upgrade_a_rejection():
    cand, baseline, seeds = _clean_case()
    llm = _fake_completion('{"verdict": "accept", "concerns": [], "critique": "looks fine"}')
    reviewer = AdversarialReviewer(completion_fn=llm)
    # Leakage forces a critical reject; the LLM's "accept" must not win.
    v = await reviewer.review(cand, baseline, seed_scores=seeds, train_query_ids={"q1"})
    assert v.verdict == "reject"


@pytest.mark.asyncio
async def test_llm_unparseable_falls_back_to_machine_verdict():
    cand, baseline, seeds = _clean_case()
    reviewer = AdversarialReviewer(completion_fn=_fake_completion("not json"))
    v = await reviewer.review(cand, baseline, seed_scores=seeds, train_query_ids={"t1"})
    assert v.verdict == "accept"  # machine verdict preserved
    assert any("unparseable" in c for c in v.concerns)


# --------------------------------------------------------------------------- #
# Lineage comparison (SPEC §8.2.1)
# --------------------------------------------------------------------------- #


def test_derive_baseline_picks_strongest_ancestor():
    weak_seed = LineageEntry(experiment_id="s1", is_seed=True, report=_report({"q1": 0.4}))
    strong_seed = LineageEntry(experiment_id="s2", is_seed=True, report=_report({"q1": 0.6}))
    parent = LineageEntry(experiment_id="p1", is_seed=False, report=_report({"q1": 0.7}))
    baseline, seeds = derive_baseline_and_seeds([weak_seed, strong_seed, parent], METRIC)
    assert baseline is parent.report  # highest primary metric
    assert sorted(seeds) == [0.4, 0.6]  # only seed entries


def test_derive_baseline_empty_lineage():
    baseline, seeds = derive_baseline_and_seeds([], METRIC)
    assert baseline is None and seeds == []


@pytest.mark.asyncio
async def test_review_from_ledger_queries_lineage():
    cand, baseline, _ = _clean_case()

    class FakeLedger:
        def __init__(self) -> None:
            self.asked: list[str] = []

        async def lineage_entries(self, experiment_id: str):
            self.asked.append(experiment_id)
            return [
                LineageEntry(experiment_id="s1", is_seed=True, report=_report({"q1": 0.49})),
                LineageEntry(experiment_id="s2", is_seed=True, report=_report({"q1": 0.50})),
                LineageEntry(experiment_id="s3", is_seed=True, report=_report({"q1": 0.51})),
                LineageEntry(experiment_id="p1", is_seed=False, report=baseline),
            ]

    ledger = FakeLedger()
    reviewer = AdversarialReviewer()
    v = await reviewer.review_from_ledger(ledger, "cand-1", cand, train_query_ids={"t1"})
    assert ledger.asked == ["cand-1"]
    assert v.verdict == "accept"


@pytest.mark.asyncio
async def test_review_with_lineage_no_baseline_is_inconclusive(reviewer):
    cand = _report({"q1": 0.9, "q2": 0.9})
    v = await reviewer.review_with_lineage(cand, [], train_query_ids={"t1"})
    # No lineage -> comparison checks cannot run -> not an accept.
    assert v.verdict == "needs_more_evidence"
    assert v.gates_pr is False


def test_custom_thresholds_tighten_regression_gate():
    strict = AdversarialReviewer(thresholds=ReviewThresholds(latency_regression_pct=0.05))
    baseline = _report({"q1": 0.5}, latency=10.0)
    cand = _report({"q1": 0.5}, latency=11.0)  # +10% now exceeds the 5% gate
    res = strict.check_regression(cand, baseline)
    assert not res.passed
