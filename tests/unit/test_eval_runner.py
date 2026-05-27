"""Tests for the evaluation runner (SPEC §9.3)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from common.schemas import Chunk, GenerationResult, GoldQuery, RetrievalCandidate
from evaluation.datasets import Dataset
from evaluation.metrics import QueryOutcome, RecallAtK
from evaluation.metrics.operational import CostPerQuery
from evaluation.runners import EvalReport, EvalRunner
from self_improvement.registry.pipeline_config import PipelineConfig


class _StubRunner:
    """Returns a perfect retrieval for every query (gold doc always at rank 1)."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.seen: list[str] = []

    async def run_query(self, query: GoldQuery, *, k: int) -> QueryOutcome:
        self.seen.append(query.query_id)
        if self.fail_on and query.query_id == self.fail_on:
            raise RuntimeError("boom")
        doc = query.relevant_doc_ids[0]
        cand = RetrievalCandidate(
            chunk=Chunk(chunk_id="c", doc_id=doc, text="t"), score=1.0, retriever="x", rank=1
        )
        gen = GenerationResult(
            text="answer", citations=[], trace_id=uuid4(), cost=0.1, tokens_in=10, tokens_out=5
        )
        return QueryOutcome(
            gold=query, candidates=[cand], generation=gen, latency_ms=5.0, cost_usd=0.1
        )


def _dataset(n: int = 3) -> Dataset:
    return Dataset(
        name="t",
        queries=[
            GoldQuery(query_id=f"q{i}", query="q", relevant_doc_ids=[f"d{i}"]) for i in range(n)
        ],
    )


@pytest.mark.asyncio
async def test_runner_aggregates_metrics():
    runner = EvalRunner(_StubRunner(), concurrency=2, k=10)
    report = await runner.run(PipelineConfig(), _dataset(3), [RecallAtK(10), CostPerQuery()])
    assert isinstance(report, EvalReport)
    assert report.n == 3
    assert report.aggregated["recall@10"] == 1.0
    assert abs(report.aggregated["cost_per_query"] - 0.1) < 1e-9
    assert len(report.per_query) == 3
    assert report.recall_at(10) == 1.0


@pytest.mark.asyncio
async def test_runner_isolates_query_failures():
    stub = _StubRunner(fail_on="q1")
    runner = EvalRunner(stub, concurrency=4, k=10)
    report = await runner.run(PipelineConfig(), _dataset(3), [RecallAtK(10)])
    # the failing query is recorded with an error and scores 0, others still run
    failed = [q for q in report.per_query if q.error]
    assert len(failed) == 1 and failed[0].query_id == "q1"
    assert report.n == 3


@pytest.mark.asyncio
async def test_report_round_trips_json():
    runner = EvalRunner(_StubRunner(), concurrency=1, k=10)
    report = await runner.run(PipelineConfig(), _dataset(2), [RecallAtK(10)])
    restored = EvalReport.model_validate_json(report.model_dump_json())
    assert restored.aggregated == report.aggregated
