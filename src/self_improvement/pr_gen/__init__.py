"""PR generation for accepted self-improvement candidates (SPEC §8.4).

Turns an accepted candidate into a human-reviewed, draft pull request with a full
evidence package — eval before/after, lineage, reviewer report, and trace links.
**No auto-merge** (anti-pattern §13): the GitHub surface here exposes no merge.
"""

from self_improvement.pr_gen.description import (
    NO_AUTO_MERGE_NOTICE,
    render_pr_description,
    render_pr_title,
)
from self_improvement.pr_gen.diff import (
    apply_pipeline_config,
    config_diff_for_candidate,
    pipeline_config_to_retrieval_block,
    render_config_update,
    unified_config_diff,
)
from self_improvement.pr_gen.evidence import (
    AcceptedCandidate,
    EvidencePackage,
    LineageEntry,
    MetricDelta,
    ReviewerReport,
    Verdict,
    build_evidence_package,
    compute_metric_deltas,
)
from self_improvement.pr_gen.generator import REVIEW_LABEL, GeneratedPR, PRGenerator
from self_improvement.pr_gen.github import (
    BranchWriter,
    GitBranchWriter,
    GitHubClient,
    GitHubRESTClient,
    OpenedPR,
    PRGenerationError,
    PROpenRequest,
    RecordingBranchWriter,
    RecordingGitHubClient,
)

__all__ = [
    # evidence
    "Verdict",
    "ReviewerReport",
    "LineageEntry",
    "MetricDelta",
    "AcceptedCandidate",
    "EvidencePackage",
    "compute_metric_deltas",
    "build_evidence_package",
    # diff
    "pipeline_config_to_retrieval_block",
    "apply_pipeline_config",
    "render_config_update",
    "unified_config_diff",
    "config_diff_for_candidate",
    # description
    "NO_AUTO_MERGE_NOTICE",
    "render_pr_title",
    "render_pr_description",
    # github
    "PRGenerationError",
    "PROpenRequest",
    "OpenedPR",
    "GitHubClient",
    "BranchWriter",
    "RecordingGitHubClient",
    "RecordingBranchWriter",
    "GitBranchWriter",
    "GitHubRESTClient",
    # generator
    "REVIEW_LABEL",
    "GeneratedPR",
    "PRGenerator",
]
