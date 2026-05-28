"""scripts/self_improve_run.py — Phase 4 self-improvement loop entry point (SPEC §14).

Composes the Phase 4 modules end-to-end: the component registry (§8.1), the
evolutionary loop (§8.2), the experiment ledger (§8.2.1), the adversarial
reviewer (§8.3), the PR generator (§8.4), and the budget guard (§8.5). Runs
under the loader's :func:`evolution_mode` so the frozen eval set (§9.1 / §13
anti-pattern) cannot be loaded during search.

Quick-start (SPEC §14):

    uv run scripts/self_improve_run.py --generations 5 --population 8 --budget-usd 50

By default the script wires a **synthetic evaluator** that scores configs from a
deterministic content hash — useful as a smoke test of the wiring on a laptop
with no infra. Swap in a real evaluator (a thin wrapper around
``evaluation.runners.EvalRunner``) by importing :func:`build_loop` from
``self_improvement.integration`` and passing your own ``evaluator`` / ``reviewer``;
the acceptance test (``tests/integration/test_phase4_acceptance.py``) does exactly
this, with mocked GitHub for the PR step.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Make ``src`` importable when invoked as a script (no editable install needed).
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from evaluation.datasets.loader import evolution_mode, load_dataset  # noqa: E402
from self_improvement.budget_guard import BudgetConfig  # noqa: E402
from self_improvement.evolutionary import Candidate  # noqa: E402
from self_improvement.integration import build_loop  # noqa: E402
from self_improvement.ledger import RunManifest  # noqa: E402
from self_improvement.registry import ComponentRegistry  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generations", type=int, default=5, help="max generations (SPEC §8.5)")
    parser.add_argument("--population", type=int, default=8, help="candidates per generation")
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=50.0,
        help="hard $ ceiling for this run (SPEC §8.5)",
    )
    parser.add_argument(
        "--max-compute-hours-per-gen",
        type=float,
        default=4.0,
        help="hard compute-hours ceiling per generation (SPEC §8.5)",
    )
    parser.add_argument(
        "--daily-usd-ceiling",
        type=float,
        default=200.0,
        help="cross-run daily $ ceiling (SPEC §8.5)",
    )
    parser.add_argument(
        "--dataset", default="dev", help="dev split name (frozen is rejected by the loader)"
    )
    parser.add_argument(
        "--rotating-dataset",
        default="rotating",
        help="rotating split shown at generation boundaries only",
    )
    parser.add_argument(
        "--ledger-root",
        default="experiments",
        help="root for runs/{run_id}/gen-NNN.jsonl + manifest.yaml (SPEC §8.2.1)",
    )
    parser.add_argument("--run-id", default=None, help="explicit run id (default: random uuid)")
    parser.add_argument(
        "--output", default="-", help="write the EvolutionReport JSON here ('-' = stdout)"
    )
    return parser.parse_args(argv)


# --- synthetic evaluator / reviewer (offline smoke-test defaults) -----------


class SyntheticEvaluator:
    """Deterministic offline evaluator: scores a config from its content hash.

    No infra required — handy for smoke-testing the wiring. Real runs swap in a
    wrapper around :class:`evaluation.runners.EvalRunner`. Metrics drift with the
    config so the loop sees a real signal, and the ``rotating_metrics`` mirror
    the dev signal so the Goodhart guard has something to inspect.
    """

    name = "synthetic"

    async def evaluate_batch(self, candidates: list[Candidate]) -> list[Candidate]:
        for c in candidates:
            digest = hashlib.sha256(c.config.model_dump_json().encode()).digest()
            # Two correlated-but-distinct signals in [0.5, 0.95].
            dev = 0.50 + (digest[0] / 255.0) * 0.45
            rot = 0.50 + (digest[1] / 255.0) * 0.45
            # ``ndcg@10`` is the loop's default ScorePolicy primary metric;
            # write it so the composite score has signal to pick a winner.
            c.metrics = {
                "recall@5": dev,
                "recall@10": min(0.99, dev + 0.05),
                "ndcg@10": dev,
            }
            c.rotating_metrics = {"recall@5": rot, "ndcg@10": rot}
            c.cost_usd = 0.01  # nominal so the budget guard accumulates
            c.compute_seconds = 0.1
        return candidates


class AcceptAllReviewer:
    """Trivial reviewer used by the smoke run. The acceptance test swaps in the
    real :class:`AdversarialReviewer` (SPEC §8.3) so the gate is actually exercised.
    """

    name = "accept_all"

    async def review_batch(self, candidates: list[Candidate]) -> list[Candidate]:
        for c in candidates:
            c.verdict = "accept"
        return candidates


# --- main --------------------------------------------------------------------


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one self-improvement loop and return its report as a JSON-able dict."""
    registry = ComponentRegistry.from_yaml()
    budget_cfg = BudgetConfig(
        max_generations=args.generations,
        max_compute_hours_per_gen=args.max_compute_hours_per_gen,
        max_usd_per_run=args.budget_usd,
        daily_usd_ceiling=args.daily_usd_ceiling,
    )
    loop, ledger, budget = build_loop(
        evaluator=SyntheticEvaluator(),
        reviewer=AcceptAllReviewer(),
        registry=registry,
        ledger_root=args.ledger_root,
        run_id=args.run_id,
        budget_config=budget_cfg,
    )
    ledger.write_manifest(
        RunManifest(
            run_id=loop.run_id,
            generations=args.generations,
            population_size=args.population,
            dataset_refs=[args.dataset, args.rotating_dataset],
            budget=budget_cfg.model_dump(),
        )
    )

    # Frozen-set isolation (SPEC §9.1, anti-pattern §13): a stray load_dataset
    # 'frozen' anywhere under this block — including inside any evaluator the
    # user wires in later — raises FrozenSetIsolationError.
    with evolution_mode():
        # Verify the dev / rotating splits at least resolve; offline runs that
        # don't actually consult the data still benefit from the early error.
        load_dataset(args.dataset)
        load_dataset(args.rotating_dataset)
        report = await loop.run(generations=args.generations, population_size=args.population)
    ledger.flush_lineage()  # best-effort denormalized graph (skips if no pyarrow)

    return {
        "run_id": report.run_id,
        "ledger_root": str(Path(args.ledger_root) / "runs" / report.run_id),
        "stopped_reason": report.stopped_reason,
        "generations_run": report.generations_run,
        "baseline_score": report.baseline_score,
        "best_score": report.best.score if report.best else None,
        "best_candidate_id": report.best.candidate_id if report.best else None,
        "best_overall_score": report.best_overall.score if report.best_overall else None,
        "budget_tripped": budget.tripped,
        "budget_trip_reason": budget.trip_reason,
        "history": report.history,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = asyncio.run(run(args))
    payload = json.dumps(summary, indent=2, default=str)
    if args.output == "-":
        print(payload)
    else:
        Path(args.output).write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
