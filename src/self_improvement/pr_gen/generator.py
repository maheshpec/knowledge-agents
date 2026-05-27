"""PR generator: turn an accepted candidate into a pull request (SPEC §8.4).

Flow for an accepted candidate:

1. **Guard** — only ``reviewer.verdict == "accept"`` proceeds; anything else
   raises (the reviewer gate, SPEC §8.3).
2. Assemble the evidence package (§8.4).
3. Compute the ``configs/default.yaml`` update + unified diff. A no-op diff
   (candidate equals current config) is refused — there is nothing to PR.
4. Create a branch, write the updated config, commit (via ``BranchWriter``).
5. Open the PR **as a draft** (via ``GitHubClient``).

**No auto-merge (SPEC §8.4 / anti-pattern §13):** the generator never merges.
The :class:`~self_improvement.pr_gen.github.GitHubClient` protocol exposes no
merge call, so this is structural, not merely conventional. The PR is opened as
a draft and labelled for human review.
"""

from __future__ import annotations

from pydantic import BaseModel

from harness.observability.logging import get_logger
from self_improvement.pr_gen.description import render_pr_description, render_pr_title
from self_improvement.pr_gen.diff import config_diff_for_candidate
from self_improvement.pr_gen.evidence import (
    AcceptedCandidate,
    EvidencePackage,
    build_evidence_package,
)
from self_improvement.pr_gen.github import (
    BranchWriter,
    GitHubClient,
    OpenedPR,
    PRGenerationError,
    PROpenRequest,
)

_log = get_logger("self_improvement.pr_gen.generator")

# Applied to every self-improvement PR so review queues can filter on it.
REVIEW_LABEL = "self-improvement"


class GeneratedPR(BaseModel):
    """Everything the generator produced: the opened PR plus its evidence/diff."""

    pr: OpenedPR
    evidence: EvidencePackage
    branch: str
    config_diff: str
    updated_config: str


class PRGenerator:
    """Generate human-reviewed PRs for accepted self-improvement candidates (§8.4)."""

    def __init__(
        self,
        github: GitHubClient,
        branch_writer: BranchWriter,
        *,
        base_branch: str = "main",
        config_path: str = "configs/default.yaml",
    ) -> None:
        self._github = github
        self._branch_writer = branch_writer
        self._base_branch = base_branch
        self._config_path = config_path

    def _branch_name(self, candidate: AcceptedCandidate) -> str:
        # Short, collision-resistant: run + experiment id. No slashes from ids.
        exp = candidate.experiment_id.replace("/", "-")
        run = candidate.run_id.replace("/", "-")
        return f"self-improve/{run}/{exp}"

    async def generate(self, candidate: AcceptedCandidate, current_config_text: str) -> GeneratedPR:
        """Open a draft PR for ``candidate``; never merges (SPEC §8.4 / §13)."""
        if candidate.reviewer.verdict != "accept":
            raise PRGenerationError(
                f"candidate {candidate.experiment_id} has verdict "
                f"'{candidate.reviewer.verdict}', not 'accept' — no PR (SPEC §8.3)"
            )

        updated_text, diff = config_diff_for_candidate(
            current_config_text, candidate.config, path=self._config_path
        )
        if not diff:
            raise PRGenerationError(
                f"candidate {candidate.experiment_id} produces no config change — nothing to PR"
            )

        evidence = build_evidence_package(candidate)
        branch = self._branch_name(candidate)

        await self._branch_writer.create_branch(branch, self._base_branch)
        await self._branch_writer.write_file(self._config_path, updated_text)
        await self._branch_writer.commit(
            f"Self-improvement: update {self._config_path} (exp {candidate.experiment_id})"
        )

        req = PROpenRequest(
            title=render_pr_title(evidence),
            body=render_pr_description(evidence),
            head_branch=branch,
            base_branch=self._base_branch,
            draft=True,  # never merge-ready without human action
            labels=[REVIEW_LABEL],
        )
        pr = await self._github.open_pull_request(req)
        _log.info(
            "pr_gen.opened",
            experiment_id=candidate.experiment_id,
            pr_number=pr.number,
            branch=branch,
            draft=pr.draft,
        )

        return GeneratedPR(
            pr=pr,
            evidence=evidence,
            branch=branch,
            config_diff=diff,
            updated_config=updated_text,
        )


__all__ = ["REVIEW_LABEL", "GeneratedPR", "PRGenerator"]
