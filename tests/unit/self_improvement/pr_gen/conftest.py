"""Shared fixtures for PR-generator tests (Phase 4R)."""

from __future__ import annotations

import pytest

from evaluation.metrics.base import MetricResult
from self_improvement.pr_gen.evidence import AcceptedCandidate, LineageEntry, ReviewerReport
from self_improvement.registry.pipeline_config import PipelineConfig

# Minimal stand-in for configs/default.yaml: enough structure that the candidate
# block is a real change. The generator only touches index.retrieval.
DEFAULT_CONFIG_TEXT = """\
llm:
  generation_model: claude-sonnet-4-6
index:
  chunker:
    name: recursive
    chunk_size: 500
  retrieval:
    retrievers: [dense, sparse_bm25]
    fusion:
      name: rrf
      rrf_k: 60
    reranker:
      name: cohere_rerank_3
      top_k: 10
    post_processors: [mmr, parent_expander]
    mmr_lambda: 0.5
    query_ops: [rewrite]
"""


def _metrics(**vals: float) -> dict[str, MetricResult]:
    return {name: MetricResult(name=name, value=v) for name, v in vals.items()}


@pytest.fixture
def default_config_text() -> str:
    return DEFAULT_CONFIG_TEXT


@pytest.fixture
def candidate_config() -> PipelineConfig:
    # Differs from the default block: adds hyde, swaps reranker, bumps rrf_k.
    return PipelineConfig(
        retrievers=["dense", "sparse_bm25"],
        fusion="rrf",
        rrf_k=80,
        reranker="voyage_rerank_2",
        reranker_top_k=12,
        post_processors=["mmr", "parent_expander"],
        mmr_lambda=0.4,
        query_ops=["rewrite", "hyde"],
    )


@pytest.fixture
def accepted_candidate(candidate_config: PipelineConfig) -> AcceptedCandidate:
    return AcceptedCandidate(
        experiment_id="exp-0007",
        run_id="run-abc",
        config=candidate_config,
        baseline_config=PipelineConfig(),
        before=_metrics(**{"ndcg@10": 0.500, "recall@20": 0.700}),
        after=_metrics(**{"ndcg@10": 0.540, "recall@20": 0.690}),
        lineage=[
            LineageEntry(experiment_id="exp-0001", generation=0, mutation_summary="seed"),
            LineageEntry(
                experiment_id="exp-0004",
                generation=2,
                mutation_summary="reranker: cohere→voyage",
                config_hash="abcdef1234567890",
            ),
        ],
        reviewer=ReviewerReport(
            verdict="accept",
            critique="Improvement is outside seed noise band; no leakage detected.",
            checks={"leakage_free": True, "above_noise_band": True, "no_latency_regression": True},
        ),
        langsmith_trace_url="https://smith.langchain.com/trace/xyz",
        heldout_results_url="https://example.com/heldout/run-abc.json",
    )
