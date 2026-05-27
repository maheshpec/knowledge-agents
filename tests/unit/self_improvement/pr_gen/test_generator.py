"""Tests for the PR generator orchestration + the no-auto-merge guarantee (SPEC §8.4 / §13)."""

from __future__ import annotations

import pytest

from self_improvement.pr_gen.generator import REVIEW_LABEL, PRGenerator
from self_improvement.pr_gen.github import (
    GitHubClient,
    PRGenerationError,
    RecordingBranchWriter,
    RecordingGitHubClient,
)


async def _generate(candidate, config_text):
    gh = RecordingGitHubClient(repo="acme/knowledge-agent")
    bw = RecordingBranchWriter()
    gen = PRGenerator(gh, bw, base_branch="main")
    result = await gen.generate(candidate, config_text)
    return gen, gh, bw, result


async def test_generate_opens_draft_pr_with_evidence(accepted_candidate, default_config_text):
    _, gh, bw, result = await _generate(accepted_candidate, default_config_text)

    # PR opened exactly once, as a draft, labelled for human review.
    assert len(gh.opened) == 1
    req = gh.opened[0]
    assert req.draft is True
    assert REVIEW_LABEL in req.labels
    assert result.pr.draft is True
    assert result.pr.number == 100
    assert "voyage_rerank_2" in result.config_diff

    # Branch created off main; updated config written + committed.
    assert bw.branch == ("self-improve/run-abc/exp-0007", "main")
    assert "configs/default.yaml" in bw.files
    assert "voyage_rerank_2" in bw.files["configs/default.yaml"]
    assert len(bw.commits) == 1


async def test_no_auto_merge_guarantee(accepted_candidate, default_config_text):
    """Structural guarantee: nothing in the GitHub surface can merge a PR."""
    _, gh, _, _ = await _generate(accepted_candidate, default_config_text)

    # 1) The client (and its protocol) expose no merge-like method at all.
    assert not any("merge" in attr.lower() for attr in dir(gh))
    assert not hasattr(GitHubClient, "merge_pull_request")

    # 2) The PR is a draft (cannot be merged without explicit human un-drafting).
    assert all(req.draft for req in gh.opened)


async def test_rejected_verdict_raises_and_opens_nothing(accepted_candidate, default_config_text):
    accepted_candidate.reviewer.verdict = "reject"
    gh = RecordingGitHubClient()
    bw = RecordingBranchWriter()
    gen = PRGenerator(gh, bw)
    with pytest.raises(PRGenerationError, match="reject"):
        await gen.generate(accepted_candidate, default_config_text)
    assert gh.opened == []  # no PR opened
    assert bw.branch is None  # no branch created


async def test_needs_more_evidence_also_blocked(accepted_candidate, default_config_text):
    accepted_candidate.reviewer.verdict = "needs_more_evidence"
    gh = RecordingGitHubClient()
    gen = PRGenerator(gh, RecordingBranchWriter())
    with pytest.raises(PRGenerationError):
        await gen.generate(accepted_candidate, default_config_text)
    assert gh.opened == []


async def test_no_op_candidate_raises(accepted_candidate, default_config_text):
    # Make the candidate identical to the current retrieval block -> empty diff.
    import yaml

    from self_improvement.registry.pipeline_config import PipelineConfig

    block = yaml.safe_load(default_config_text)["index"]["retrieval"]
    accepted_candidate.config = PipelineConfig(
        retrievers=block["retrievers"],
        fusion=block["fusion"]["name"],
        rrf_k=block["fusion"]["rrf_k"],
        reranker=block["reranker"]["name"],
        reranker_top_k=block["reranker"]["top_k"],
        post_processors=block["post_processors"],
        mmr_lambda=block["mmr_lambda"],
        query_ops=block["query_ops"],
    )
    gh = RecordingGitHubClient()
    gen = PRGenerator(gh, RecordingBranchWriter())
    with pytest.raises(PRGenerationError, match="no config change"):
        await gen.generate(accepted_candidate, default_config_text)
    assert gh.opened == []


async def test_branch_name_sanitizes_ids(accepted_candidate, default_config_text):
    accepted_candidate.experiment_id = "exp/weird/id"
    _, _, bw, result = await _generate(accepted_candidate, default_config_text)
    assert "/" not in result.branch.split("self-improve/")[1].split("/")[-1]
    assert result.branch == "self-improve/run-abc/exp-weird-id"
