"""Tests for config-diff generation (SPEC §8.4)."""

from __future__ import annotations

import yaml

from self_improvement.pr_gen.diff import (
    apply_pipeline_config,
    config_diff_for_candidate,
    pipeline_config_to_retrieval_block,
    render_config_update,
)
from self_improvement.registry.pipeline_config import PipelineConfig


def test_block_maps_all_config_fields():
    cfg = PipelineConfig(rrf_k=80, reranker="voyage_rerank_2", reranker_top_k=12, mmr_lambda=0.4)
    block = pipeline_config_to_retrieval_block(cfg)
    assert block["fusion"] == {"name": "rrf", "rrf_k": 80}
    assert block["reranker"] == {"name": "voyage_rerank_2", "top_k": 12}
    assert block["mmr_lambda"] == 0.4
    assert block["query_ops"] == cfg.query_ops


def test_apply_does_not_mutate_input(candidate_config):
    current = {"index": {"retrieval": {"retrievers": ["dense"]}}, "llm": {"x": 1}}
    updated = apply_pipeline_config(current, candidate_config)
    assert current["index"]["retrieval"] == {"retrievers": ["dense"]}  # untouched
    assert updated["index"]["retrieval"]["reranker"]["name"] == "voyage_rerank_2"
    assert updated["llm"] == {"x": 1}  # unrelated keys preserved


def test_apply_creates_index_when_missing(candidate_config):
    updated = apply_pipeline_config({}, candidate_config)
    assert "retrieval" in updated["index"]


def test_render_update_roundtrips_and_changes_retrieval(default_config_text, candidate_config):
    updated_text = render_config_update(default_config_text, candidate_config)
    parsed = yaml.safe_load(updated_text)
    # Unrelated keys survive.
    assert parsed["llm"]["generation_model"] == "claude-sonnet-4-6"
    assert parsed["index"]["chunker"]["chunk_size"] == 500
    # Retrieval block reflects the candidate.
    assert parsed["index"]["retrieval"]["fusion"]["rrf_k"] == 80
    assert parsed["index"]["retrieval"]["query_ops"] == ["rewrite", "hyde"]


def test_diff_is_unified_and_nonempty(default_config_text, candidate_config):
    updated_text, diff = config_diff_for_candidate(
        default_config_text, candidate_config, path="configs/default.yaml"
    )
    assert "a/configs/default.yaml" in diff
    assert "b/configs/default.yaml" in diff
    assert "voyage_rerank_2" in diff  # the change shows up
    assert diff.startswith("---")


def test_identical_config_yields_empty_diff(default_config_text):
    # A candidate equal to the current block produces no diff.
    current = yaml.safe_load(default_config_text)
    block = current["index"]["retrieval"]
    same = PipelineConfig(
        retrievers=block["retrievers"],
        fusion=block["fusion"]["name"],
        rrf_k=block["fusion"]["rrf_k"],
        reranker=block["reranker"]["name"],
        reranker_top_k=block["reranker"]["top_k"],
        post_processors=block["post_processors"],
        mmr_lambda=block["mmr_lambda"],
        query_ops=block["query_ops"],
    )
    _, diff = config_diff_for_candidate(default_config_text, same)
    assert diff == ""  # semantically identical -> no diff (formatting ignored)
