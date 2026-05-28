"""End-to-end evaluation test (SPEC §9, epic ka-5ps acceptance).

Runs the offline eval pipeline over the seed dev set and asserts the acceptance
bar: a complete EvalReport with recall@10 >= 0.7. Offline (hash embedder + BM25 +
stub draft), so it needs no API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ is not a package; import eval_run by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import eval_run  # noqa: E402


@pytest.mark.xfail(
    reason="Spec-scale recall gap tracked as ka-8t7 (Mayor decision on ka-94g): the "
    "expanded dev set surfaces a real retrieval-quality gap the self-improvement "
    "loop will own closing. The 0.7 bar stays as the honest target — do not lower "
    "it (Goodhart anti-pattern §13) — just xfail until ka-8t7 lands.",
    strict=False,
)
@pytest.mark.asyncio
async def test_offline_dev_eval_meets_recall_bar():
    args = eval_run._parse_args(["--dataset", "dev", "--offline"])
    report = await eval_run.run_eval(args)

    assert report.n >= 50
    assert report.dataset == "dev"
    # complete report: every declared metric is present in the aggregate
    for metric in ("recall@5", "recall@10", "recall@20", "mrr", "hit_rate", "ndcg@10"):
        assert metric in report.aggregated
    # acceptance: recall@10 on dev >= 0.7 with the default (offline) pipeline
    assert report.recall_at(10) >= 0.7
    assert len(report.per_query) == report.n


@pytest.mark.asyncio
async def test_smoke_subset_runs():
    args = eval_run._parse_args(["--dataset", "dev", "--offline", "--smoke", "20"])
    report = await eval_run.run_eval(args)
    assert report.n == 20
