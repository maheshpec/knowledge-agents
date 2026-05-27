"""Tests for evidence-package assembly (SPEC §8.4)."""

from __future__ import annotations

import pytest

from evaluation.metrics.base import MetricResult
from self_improvement.pr_gen.description import render_pr_description, render_pr_title
from self_improvement.pr_gen.evidence import (
    MetricDelta,
    build_evidence_package,
    compute_metric_deltas,
)


def test_metric_delta_math():
    d = MetricDelta(name="ndcg@10", before=0.5, after=0.54)
    assert d.delta == pytest.approx(0.04)
    assert d.pct_change == pytest.approx(0.08)
    assert d.improved is True


def test_metric_delta_zero_baseline_pct_is_none():
    d = MetricDelta(name="x", before=0.0, after=0.3)
    assert d.pct_change is None
    assert d.improved is True


def test_compute_deltas_unions_keys():
    before = {"a": MetricResult(name="a", value=1.0)}
    after = {"a": MetricResult(name="a", value=2.0), "b": MetricResult(name="b", value=0.5)}
    deltas = {d.name: d for d in compute_metric_deltas(before, after)}
    assert deltas["a"].before == 1.0 and deltas["a"].after == 2.0
    assert deltas["b"].before == 0.0 and deltas["b"].after == 0.5  # missing-before -> 0


def test_build_package_carries_links_and_splits_improved(accepted_candidate):
    pkg = build_evidence_package(accepted_candidate)
    assert pkg.langsmith_trace_url == "https://smith.langchain.com/trace/xyz"
    assert pkg.heldout_results_url == "https://example.com/heldout/run-abc.json"
    improved = {d.name for d in pkg.improved_metrics()}
    regressed = {d.name for d in pkg.regressed_metrics()}
    assert "ndcg@10" in improved  # 0.50 -> 0.54
    assert "recall@20" in regressed  # 0.70 -> 0.69


def test_description_contains_all_evidence_sections(accepted_candidate):
    pkg = build_evidence_package(accepted_candidate)
    body = render_pr_description(pkg)
    # Each required evidence element from SPEC §8.4 appears.
    assert "Eval metrics" in body
    assert "ndcg@10" in body and "0.5000" in body and "0.5400" in body
    assert "Candidate lineage" in body and "exp-0004" in body
    assert "reviewer-cohere" not in body  # sanity: no leakage of unrelated text
    assert "Adversarial reviewer report" in body and "accept" in body
    assert "smith.langchain.com/trace/xyz" in body
    assert "heldout/run-abc.json" in body
    # No-auto-merge notice present (twice: header + footer).
    assert body.count("do NOT auto-merge") >= 1


def test_title_highlights_best_metric(accepted_candidate):
    pkg = build_evidence_package(accepted_candidate)
    title = render_pr_title(pkg)
    assert "ndcg@10" in title and "exp-0007" in title
