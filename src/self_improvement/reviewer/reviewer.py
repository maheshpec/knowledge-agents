"""Adversarial reviewer (SPEC §8.3).

A separate, LLM-driven gate whose explicit job is to find reasons a candidate
result is *invalid* — the guard against the "overexcitement" failure mode of
autonomous research loops. It runs four deterministic checks against a candidate
evaluation and its lineage (SPEC §8.2.1):

1. **Leakage** — the eval set must not overlap the training/seed queries.
2. **Narrow slice** — a gain carried by too few queries is unproven.
3. **Noise band** — the gain must exceed seed-variance, not ride on it.
4. **Regression** — latency/cost must not regress beyond threshold.

Each check yields a :class:`CheckResult`; a *critical* failure (leakage, noise,
regression) forces ``reject``, a non-critical failure (narrow slice, or any check
that lacks a baseline to judge) demotes to ``needs_more_evidence``. An injectable
``completion_fn`` (mirroring the enrichers, SPEC §7.3) then reviews the evidence
adversarially and may *only make the verdict more conservative* — it can raise
concerns or downgrade ``accept``→``needs_more_evidence`` but never upgrade a
machine-determined failure. The final ``verdict`` gates PR creation (SPEC §8.4).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Awaitable, Callable

from evaluation.runners.runner import EvalReport
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from self_improvement.reviewer.models import (
    CheckResult,
    LineageEntry,
    LineageProvider,
    ReviewerVerdict,
    ReviewThresholds,
)

_log = get_logger("self_improvement.reviewer")

# Prompt-in, text-out — same shape as the query-op completer; injected for tests.
CompletionFn = Callable[[str], Awaitable[str]]

DEFAULT_REVIEWER_MODEL = "claude-haiku-4-5-20251001"

# Verdict severity ordering: the reviewer always keeps the most severe verdict
# between the deterministic checks and the LLM (the LLM cannot upgrade).
_SEVERITY = {"accept": 0, "needs_more_evidence": 1, "reject": 2}
_BY_SEVERITY = {v: k for k, v in _SEVERITY.items()}

REVIEW_PROMPT = """You are an adversarial reviewer of an automated retrieval-pipeline \
experiment. Your job is to find reasons the claimed improvement is INVALID or \
overstated — be skeptical, not generous.

Deterministic checks already ran (verdict so far: {machine_verdict}):
{checks}

Return ONLY a JSON object (no prose, no code fences):
  {{"verdict": "accept|reject|needs_more_evidence",
    "concerns": ["..."],
    "critique": "<2-3 sentence assessment>"}}

You may keep or make the verdict MORE conservative (accept -> needs_more_evidence \
-> reject); never upgrade it past what the checks allow. List concrete concerns."""


def default_completion_fn(model: str = DEFAULT_REVIEWER_MODEL) -> CompletionFn:
    """Build a ``str -> str`` async completer backed by ChatAnthropic (lazy import)."""

    async def _complete(prompt: str) -> str:
        from typing import Any

        from langchain_anthropic import ChatAnthropic

        from common.settings import get_settings

        init_kwargs: dict[str, Any] = {
            "model": model,
            "api_key": get_settings().anthropic_api_key,
            "max_tokens": 1024,
        }
        llm = ChatAnthropic(**init_kwargs)
        response = await llm.ainvoke(prompt)
        content = response.content
        return content if isinstance(content, str) else str(content)

    return _complete


# --------------------------------------------------------------------------- #
# Report accessors
# --------------------------------------------------------------------------- #


def _primary(report: EvalReport, metric: str) -> float:
    return report.aggregated.get(metric, 0.0)


def _per_query_primary(report: EvalReport, metric: str) -> dict[str, float]:
    return {qr.query_id: qr.metrics.get(metric, 0.0) for qr in report.per_query}


def _mean_latency(report: EvalReport) -> float:
    vals = [qr.latency_ms for qr in report.per_query]
    return statistics.fmean(vals) if vals else 0.0


def _mean_cost(report: EvalReport) -> float:
    vals = [qr.cost_usd for qr in report.per_query]
    return statistics.fmean(vals) if vals else 0.0


def _pct_increase(baseline: float, candidate: float) -> float:
    """Relative increase of candidate over baseline; 0 when baseline is 0."""
    if baseline <= 0.0:
        return 0.0
    return (candidate - baseline) / baseline


# --------------------------------------------------------------------------- #
# Lineage helpers (SPEC §8.2.1 comparison)
# --------------------------------------------------------------------------- #


def derive_baseline_and_seeds(
    lineage: list[LineageEntry], metric: str
) -> tuple[EvalReport | None, list[float]]:
    """From a candidate's lineage, pick the baseline to beat and the seed scores.

    The *baseline* is the strongest ancestor (highest ``metric``) — a candidate
    must outperform the best thing already in its lineage, not a weak one. The
    *seed scores* are the ``metric`` values of every seed ancestor; their spread
    defines the noise band (SPEC §8.2.1 replayability / §8.3 noise check).
    """
    if not lineage:
        return None, []
    baseline = max(lineage, key=lambda e: _primary(e.report, metric)).report
    seed_scores = [_primary(e.report, metric) for e in lineage if e.is_seed]
    return baseline, seed_scores


class AdversarialReviewer:
    """The §8.3 gate: deterministic checks + an adversarial LLM pass."""

    def __init__(
        self,
        completion_fn: CompletionFn | None = None,
        *,
        thresholds: ReviewThresholds | None = None,
    ) -> None:
        self._complete = completion_fn
        self.thresholds = thresholds or ReviewThresholds()

    # --- individual checks (pure, synchronous, unit-testable) ---

    def check_leakage(self, candidate: EvalReport, train_query_ids: set[str]) -> CheckResult:
        """Eval set must share no query ids with the training/seed configs."""
        eval_ids = {qr.query_id for qr in candidate.per_query}
        overlap = sorted(eval_ids & set(train_query_ids))
        passed = not overlap
        summary = (
            "no overlap between eval and training/seed queries"
            if passed
            else f"{len(overlap)} eval query(ies) leaked from training/seed set"
        )
        return CheckResult(
            name="leakage_free",
            passed=passed,
            critical=True,
            summary=summary,
            detail={"overlap": overlap, "n_overlap": len(overlap), "n_eval": len(eval_ids)},
        )

    def check_narrow_slice(self, candidate: EvalReport, baseline: EvalReport | None) -> CheckResult:
        """A net gain must be spread across queries, not confined to a few."""
        t = self.thresholds
        if baseline is None:
            return CheckResult(
                name="broad_improvement",
                passed=False,
                critical=False,
                summary="no baseline in lineage; cannot confirm the gain is broad",
                detail={},
            )
        cand_pq = _per_query_primary(candidate, t.primary_metric)
        base_pq = _per_query_primary(baseline, t.primary_metric)
        shared = sorted(set(cand_pq) & set(base_pq))
        improved = [q for q in shared if cand_pq[q] - base_pq[q] > t.epsilon]
        regressed = [q for q in shared if base_pq[q] - cand_pq[q] > t.epsilon]
        net = _primary(candidate, t.primary_metric) - _primary(baseline, t.primary_metric)
        frac = (len(improved) / len(shared)) if shared else 0.0
        # Only a *positive* net gain can be "narrow"; no gain is the other checks' job.
        narrow = net > t.epsilon and frac < t.min_improved_fraction
        passed = not narrow
        summary = (
            f"gain spread across {len(improved)}/{len(shared)} shared queries"
            if passed
            else (
                f"gain confined to {len(improved)}/{len(shared)} queries "
                f"({frac:.0%} < {t.min_improved_fraction:.0%} required)"
            )
        )
        return CheckResult(
            name="broad_improvement",
            passed=passed,
            critical=False,
            summary=summary,
            detail={
                "net_delta": net,
                "improved": len(improved),
                "regressed": len(regressed),
                "shared": len(shared),
                "improved_fraction": frac,
            },
        )

    def check_noise_band(
        self,
        candidate: EvalReport,
        baseline: EvalReport | None,
        seed_scores: list[float],
    ) -> CheckResult:
        """The gain over baseline must exceed the seed-variance noise band."""
        t = self.thresholds
        if baseline is None:
            return CheckResult(
                name="exceeds_noise_band",
                passed=False,
                critical=False,
                summary="no baseline in lineage; cannot size the improvement",
                detail={},
            )
        if len(seed_scores) < 2:
            return CheckResult(
                name="exceeds_noise_band",
                passed=False,
                critical=False,
                summary=f"only {len(seed_scores)} seed sample(s); cannot estimate noise band",
                detail={"n_seed": len(seed_scores)},
            )
        improvement = _primary(candidate, t.primary_metric) - _primary(baseline, t.primary_metric)
        seed_std = statistics.stdev(seed_scores)
        band = t.noise_band_sigmas * seed_std
        passed = improvement > band
        summary = (
            f"improvement {improvement:.4f} exceeds noise band {band:.4f}"
            if passed
            else f"improvement {improvement:.4f} within noise band {band:.4f} (not significant)"
        )
        return CheckResult(
            name="exceeds_noise_band",
            passed=passed,
            critical=True,
            summary=summary,
            detail={
                "improvement": improvement,
                "noise_band": band,
                "seed_std": seed_std,
                "sigmas": t.noise_band_sigmas,
                "n_seed": len(seed_scores),
            },
        )

    def check_regression(self, candidate: EvalReport, baseline: EvalReport | None) -> CheckResult:
        """Latency and cost must not regress beyond the configured thresholds."""
        t = self.thresholds
        if baseline is None:
            return CheckResult(
                name="no_perf_regression",
                passed=False,
                critical=False,
                summary="no baseline in lineage; cannot check for regression",
                detail={},
            )
        lat_pct = _pct_increase(_mean_latency(baseline), _mean_latency(candidate))
        cost_pct = _pct_increase(_mean_cost(baseline), _mean_cost(candidate))
        lat_bad = lat_pct > t.latency_regression_pct
        cost_bad = cost_pct > t.cost_regression_pct
        passed = not (lat_bad or cost_bad)
        if passed:
            summary = f"latency {lat_pct:+.0%}, cost {cost_pct:+.0%} within thresholds"
        else:
            parts = []
            if lat_bad:
                parts.append(f"latency +{lat_pct:.0%} > {t.latency_regression_pct:.0%}")
            if cost_bad:
                parts.append(f"cost +{cost_pct:.0%} > {t.cost_regression_pct:.0%}")
            summary = "regression: " + "; ".join(parts)
        return CheckResult(
            name="no_perf_regression",
            passed=passed,
            critical=True,
            summary=summary,
            detail={
                "latency_pct": lat_pct,
                "cost_pct": cost_pct,
                "latency_threshold": t.latency_regression_pct,
                "cost_threshold": t.cost_regression_pct,
            },
        )

    def run_checks(
        self,
        candidate: EvalReport,
        baseline: EvalReport | None,
        *,
        seed_scores: list[float],
        train_query_ids: set[str],
    ) -> list[CheckResult]:
        """Run the full check battery (sync; the LLM pass is separate)."""
        return [
            self.check_leakage(candidate, train_query_ids),
            self.check_narrow_slice(candidate, baseline),
            self.check_noise_band(candidate, baseline, seed_scores),
            self.check_regression(candidate, baseline),
        ]

    # --- verdict ---

    @traced(span_name="self_improvement.reviewer.review")
    async def review(
        self,
        candidate: EvalReport,
        baseline: EvalReport | None = None,
        *,
        seed_scores: list[float] | None = None,
        train_query_ids: set[str] | None = None,
    ) -> ReviewerVerdict:
        """Score a candidate and return a gated verdict (SPEC §8.3)."""
        checks = self.run_checks(
            candidate,
            baseline,
            seed_scores=seed_scores or [],
            train_query_ids=train_query_ids or set(),
        )
        machine_verdict = _derive_verdict(checks)
        concerns = [c.summary for c in checks if not c.passed]

        verdict = machine_verdict
        critique = _deterministic_critique(checks, machine_verdict)
        if self._complete is not None:
            llm_verdict, llm_concerns, llm_critique = await self._adversarial_pass(
                checks, machine_verdict
            )
            # The LLM may only tighten the gate, never loosen it.
            verdict = _BY_SEVERITY[max(_SEVERITY[machine_verdict], _SEVERITY[llm_verdict])]
            concerns = _dedupe(concerns + llm_concerns)
            if llm_critique:
                critique = llm_critique

        _log.info(
            "reviewer.verdict",
            verdict=verdict,
            machine_verdict=machine_verdict,
            failed=[c.name for c in checks if not c.passed],
        )
        return ReviewerVerdict(verdict=verdict, critique=critique, checks=checks, concerns=concerns)

    async def review_with_lineage(
        self,
        candidate: EvalReport,
        lineage: list[LineageEntry],
        *,
        train_query_ids: set[str] | None = None,
    ) -> ReviewerVerdict:
        """Review a candidate against its lineage entries (SPEC §8.2.1)."""
        baseline, seed_scores = derive_baseline_and_seeds(lineage, self.thresholds.primary_metric)
        return await self.review(
            candidate,
            baseline,
            seed_scores=seed_scores,
            train_query_ids=train_query_ids,
        )

    async def review_from_ledger(
        self,
        provider: LineageProvider,
        experiment_id: str,
        candidate: EvalReport,
        *,
        train_query_ids: set[str] | None = None,
    ) -> ReviewerVerdict:
        """Query the §8.2.1 ledger for the candidate's lineage, then review."""
        lineage = await provider.lineage_entries(experiment_id)
        return await self.review_with_lineage(candidate, lineage, train_query_ids=train_query_ids)

    # --- internals ---

    async def _adversarial_pass(
        self, checks: list[CheckResult], machine_verdict: str
    ) -> tuple[str, list[str], str]:
        prompt = REVIEW_PROMPT.format(
            machine_verdict=machine_verdict, checks=_format_checks(checks)
        )
        try:
            raw = await self._complete(prompt)  # type: ignore[misc]
            data = json.loads(_strip_code_fence(raw))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            _log.warning("reviewer.llm_unparseable", error=str(exc))
            return machine_verdict, ["adversarial reviewer response was unparseable"], ""
        verdict = data.get("verdict")
        if verdict not in _SEVERITY:
            verdict = machine_verdict
        concerns = [str(c) for c in data.get("concerns", []) if c]
        critique = str(data.get("critique", ""))
        return verdict, concerns, critique


def _derive_verdict(checks: list[CheckResult]) -> str:
    """Reject on any critical failure; else needs_more_evidence on any failure."""
    if any(not c.passed and c.critical for c in checks):
        return "reject"
    if any(not c.passed for c in checks):
        return "needs_more_evidence"
    return "accept"


def _deterministic_critique(checks: list[CheckResult], verdict: str) -> str:
    failed = [c for c in checks if not c.passed]
    if not failed:
        return "All adversarial checks passed; improvement looks valid."
    bullets = "; ".join(f"{c.name}: {c.summary}" for c in failed)
    return f"Verdict {verdict}. Failing checks — {bullets}."


def _format_checks(checks: list[CheckResult]) -> str:
    lines = []
    for c in checks:
        flag = "PASS" if c.passed else ("FAIL/critical" if c.critical else "FAIL")
        lines.append(f"- [{flag}] {c.name}: {c.summary} {c.detail}")
    return "\n".join(lines)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


__all__ = [
    "AdversarialReviewer",
    "CompletionFn",
    "DEFAULT_REVIEWER_MODEL",
    "REVIEW_PROMPT",
    "default_completion_fn",
    "derive_baseline_and_seeds",
]
