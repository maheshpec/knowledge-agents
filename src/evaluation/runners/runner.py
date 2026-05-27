"""Evaluation runner (SPEC §9.3).

:class:`EvalRunner` executes a :class:`Dataset` through a pipeline, scores every
query with the supplied metrics, and folds the results into an :class:`EvalReport`
(per-query + aggregated + lineage + optional LangSmith trace URL). Queries run
concurrently under a semaphore cap so a large dataset doesn't fan out unbounded.

How a query is executed is abstracted behind :class:`PipelineRunner`. The default
:class:`OrchestratorPipelineRunner` drives a compiled LangGraph orchestrator app
and reads its final-state ``candidates`` + ``result``; tests inject a stub runner.
The ``PipelineConfig`` genome is recorded in the report's lineage and used to log
the run as a comparable LangSmith experiment.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from common.schemas import GoldQuery
from evaluation.datasets.loader import Dataset
from evaluation.metrics.base import MeanMetric, MetricResult, QueryOutcome
from self_improvement.registry.pipeline_config import PipelineConfig


@runtime_checkable
class PipelineRunner(Protocol):
    """Turns a gold query into a scored :class:`QueryOutcome` (candidates + answer)."""

    async def run_query(self, query: GoldQuery, *, k: int) -> QueryOutcome: ...


class QueryReport(BaseModel):
    """Per-query slice of an :class:`EvalReport`."""

    query_id: str
    metrics: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    num_candidates: int = 0
    num_citations: int = 0
    error: str | None = None


class EvalReport(BaseModel):
    """Full evaluation result: aggregated + per-query + lineage (SPEC §9.3)."""

    dataset: str
    n: int
    aggregated: dict[str, float] = Field(default_factory=dict)
    per_query: list[QueryReport] = Field(default_factory=list)
    pipeline_config: PipelineConfig | None = None
    metrics: list[str] = Field(default_factory=list)
    started_at: str = ""
    duration_s: float = 0.0
    langsmith_url: str | None = None

    def recall_at(self, k: int) -> float:
        """Convenience accessor for ``recall@k`` (used by the CI regression gate)."""
        return self.aggregated.get(f"recall@{k}", 0.0)


class EvalRunner:
    """Runs a dataset through a pipeline and scores it (SPEC §9.3)."""

    def __init__(
        self,
        runner: PipelineRunner,
        *,
        concurrency: int = 8,
        k: int = 20,
        log_to_langsmith: bool = False,
        langsmith_project: str = "knowledge-agent",
    ) -> None:
        self.runner = runner
        self.concurrency = max(1, concurrency)
        self.k = k
        self.log_to_langsmith = log_to_langsmith
        self.langsmith_project = langsmith_project

    async def _score_query(
        self, query: GoldQuery, metrics: list[MeanMetric]
    ) -> tuple[QueryReport, list[MetricResult]]:
        try:
            outcome = await self.runner.run_query(query, k=self.k)
        except Exception as exc:
            empty = QueryOutcome(gold=query, error=str(exc))
            outcome = empty
        results = [m.compute(outcome) for m in metrics]
        report = QueryReport(
            query_id=query.query_id,
            metrics={r.name: r.value for r in results},
            latency_ms=outcome.latency_ms,
            cost_usd=outcome.cost_usd,
            num_candidates=len(outcome.candidates),
            num_citations=len(outcome.generation.citations) if outcome.generation else 0,
            error=outcome.error,
        )
        return report, results

    async def run(
        self,
        pipeline_config: PipelineConfig,
        dataset: Dataset,
        metrics: list[MeanMetric],
    ) -> EvalReport:
        """Execute ``dataset`` through the pipeline and score it (SPEC §9.3)."""
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        sem = asyncio.Semaphore(self.concurrency)

        async def _guarded(q: GoldQuery) -> tuple[QueryReport, list[MetricResult]]:
            async with sem:
                return await self._score_query(q, metrics)

        scored = await asyncio.gather(*[_guarded(q) for q in dataset.queries])
        per_query = [s[0] for s in scored]

        # Aggregate each metric across all per-query results.
        aggregated: dict[str, float] = {}
        for i, metric in enumerate(metrics):
            agg = metric.aggregate([s[1][i] for s in scored])
            aggregated[agg.name] = agg.value

        report = EvalReport(
            dataset=dataset.name,
            n=len(dataset),
            aggregated=aggregated,
            per_query=per_query,
            pipeline_config=pipeline_config,
            metrics=[m.name for m in metrics],
            started_at=started.isoformat(),
            duration_s=round(time.perf_counter() - t0, 3),
        )
        if self.log_to_langsmith:
            report.langsmith_url = _log_langsmith(report, self.langsmith_project)
        return report


def _log_langsmith(report: EvalReport, project: str) -> str | None:
    """Log the run as a LangSmith experiment; return its URL (best effort).

    Imported lazily and wrapped defensively: a missing dep or unset API key must
    never fail an evaluation run.
    """
    try:  # pragma: no cover - requires langsmith + network
        from langsmith import Client

        client = Client()
        run = client.create_run(
            name=f"eval:{report.dataset}",
            run_type="chain",
            project_name=project,
            inputs={
                "dataset": report.dataset,
                "n": report.n,
                "pipeline_config": report.pipeline_config.model_dump()
                if report.pipeline_config
                else None,
            },
            outputs=report.aggregated,
        )
        return f"https://smith.langchain.com/o/-/projects/p/{project}?run={getattr(run, 'id', '')}"
    except Exception:
        return None


__all__ = [
    "EvalRunner",
    "EvalReport",
    "QueryReport",
    "PipelineRunner",
]
