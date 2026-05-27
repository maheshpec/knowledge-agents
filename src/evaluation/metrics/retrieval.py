"""Retrieval metrics (SPEC §9.2): Recall@k, Precision@k, nDCG@k, MRR, HitRate.

Relevance is matched at two granularities so gold labels stay robust to chunker
configuration: a candidate counts as relevant if its ``chunk_id`` is in
``relevant_chunk_ids`` **or** its ``doc_id`` is in ``relevant_doc_ids``. The
recall denominator is the number of distinct gold targets (chunk ids + doc ids),
so a doc-level gold label needs only one matching chunk to be fully recalled.
"""

from __future__ import annotations

import math

from common.schemas import GoldQuery, RetrievalCandidate
from evaluation.metrics.base import MeanMetric, MetricResult, QueryOutcome


def _is_relevant(candidate: RetrievalCandidate, gold_chunks: set[str], gold_docs: set[str]) -> bool:
    return candidate.chunk.chunk_id in gold_chunks or candidate.chunk.doc_id in gold_docs


def _gold_targets(gold: GoldQuery) -> tuple[set[str], set[str]]:
    return set(gold.relevant_chunk_ids), set(gold.relevant_doc_ids)


def _hit_targets(
    candidates: list[RetrievalCandidate], gold_chunks: set[str], gold_docs: set[str]
) -> tuple[set[str], set[str]]:
    """Which distinct gold chunk/doc targets were retrieved among ``candidates``."""
    hit_chunks = {c.chunk.chunk_id for c in candidates if c.chunk.chunk_id in gold_chunks}
    hit_docs = {c.chunk.doc_id for c in candidates if c.chunk.doc_id in gold_docs}
    return hit_chunks, hit_docs


class RecallAtK(MeanMetric):
    """Fraction of distinct gold targets retrieved within the top ``k``."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.name = f"recall@{k}"

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        gold_chunks, gold_docs = _gold_targets(outcome.gold)
        total = len(gold_chunks) + len(gold_docs)
        if total == 0:
            return MetricResult(name=self.name, value=0.0, detail={"total_relevant": 0})
        topk = outcome.candidates[: self.k]
        hit_chunks, hit_docs = _hit_targets(topk, gold_chunks, gold_docs)
        value = (len(hit_chunks) + len(hit_docs)) / total
        return MetricResult(
            name=self.name,
            value=value,
            detail={"hits": len(hit_chunks) + len(hit_docs), "total_relevant": total},
        )


class PrecisionAtK(MeanMetric):
    """Fraction of the top ``k`` retrieved candidates that are relevant."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.name = f"precision@{k}"

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        gold_chunks, gold_docs = _gold_targets(outcome.gold)
        topk = outcome.candidates[: self.k]
        if not topk:
            return MetricResult(name=self.name, value=0.0, detail={"retrieved": 0})
        relevant = sum(1 for c in topk if _is_relevant(c, gold_chunks, gold_docs))
        return MetricResult(
            name=self.name,
            value=relevant / len(topk),
            detail={"relevant": relevant, "retrieved": len(topk)},
        )


class NDCGAtK(MeanMetric):
    """Normalized discounted cumulative gain at ``k`` (binary relevance)."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.name = f"ndcg@{k}"

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        gold_chunks, gold_docs = _gold_targets(outcome.gold)
        total = len(gold_chunks) + len(gold_docs)
        if total == 0:
            return MetricResult(name=self.name, value=0.0)
        topk = outcome.candidates[: self.k]
        dcg = 0.0
        for i, c in enumerate(topk):
            if _is_relevant(c, gold_chunks, gold_docs):
                dcg += 1.0 / math.log2(i + 2)  # rank i is 0-based; +2 -> log2(2) at rank 0
        ideal_hits = min(total, self.k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        value = dcg / idcg if idcg > 0 else 0.0
        return MetricResult(name=self.name, value=value, detail={"dcg": dcg, "idcg": idcg})


class MRR(MeanMetric):
    """Mean reciprocal rank of the first relevant candidate."""

    name = "mrr"

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        gold_chunks, gold_docs = _gold_targets(outcome.gold)
        for i, c in enumerate(outcome.candidates):
            if _is_relevant(c, gold_chunks, gold_docs):
                return MetricResult(name=self.name, value=1.0 / (i + 1), detail={"rank": i + 1})
        return MetricResult(name=self.name, value=0.0, detail={"rank": None})


class HitRate(MeanMetric):
    """1.0 if any relevant candidate is retrieved within ``k``, else 0.0."""

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.name = "hit_rate"

    def compute(self, outcome: QueryOutcome) -> MetricResult:
        gold_chunks, gold_docs = _gold_targets(outcome.gold)
        topk = outcome.candidates[: self.k]
        hit = any(_is_relevant(c, gold_chunks, gold_docs) for c in topk)
        return MetricResult(name=self.name, value=1.0 if hit else 0.0)


def default_retrieval_metrics() -> list[MeanMetric]:
    """The canonical retrieval metric suite from ``configs/eval.yaml``."""
    return [
        RecallAtK(5),
        RecallAtK(10),
        RecallAtK(20),
        PrecisionAtK(10),
        NDCGAtK(10),
        MRR(),
        HitRate(10),
    ]


__all__ = [
    "RecallAtK",
    "PrecisionAtK",
    "NDCGAtK",
    "MRR",
    "HitRate",
    "default_retrieval_metrics",
]
