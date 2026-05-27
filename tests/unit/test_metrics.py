"""Unit tests for evaluation metrics (SPEC §9.2, epic ka-5ps acceptance).

Each metric is checked against a hand-built ground-truth case with a known value.
"""

from __future__ import annotations

import math

from common.schemas import Chunk, Citation, GenerationResult, GoldQuery, RetrievalCandidate, Source
from evaluation.metrics import (
    MRR,
    CitationPrecision,
    CitationRecall,
    CostPerQuery,
    HitRate,
    LatencyP50,
    LatencyP95,
    LexicalOverlapJudge,
    NDCGAtK,
    PrecisionAtK,
    QueryOutcome,
    RecallAtK,
    TokenEfficiency,
)


def _cand(
    chunk_id: str, doc_id: str = "d", text: str = "body", rank: int = 1
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text),
        score=1.0 / rank,
        retriever="dense",
        rank=rank,
    )


def _outcome(candidate_ids: list[str], relevant_ids: list[str], **kw) -> QueryOutcome:
    gold = GoldQuery(query_id="q1", query="q", relevant_chunk_ids=relevant_ids)
    cands = [_cand(cid, rank=i + 1) for i, cid in enumerate(candidate_ids)]
    return QueryOutcome(gold=gold, candidates=cands, **kw)


# --- retrieval metrics ------------------------------------------------------


def test_recall_at_k():
    # gold = {c1, c3}; top-2 retrieves c1 only -> 1/2
    outcome = _outcome(["c1", "c9", "c3"], ["c1", "c3"])
    assert RecallAtK(2).compute(outcome).value == 0.5
    assert RecallAtK(3).compute(outcome).value == 1.0


def test_precision_at_k():
    # top-4 has 2 relevant (c1, c3) of 4 -> 0.5
    outcome = _outcome(["c1", "x", "c3", "y"], ["c1", "c3"])
    assert PrecisionAtK(4).compute(outcome).value == 0.5


def test_ndcg_at_k():
    # relevant at ranks 1 and 3 (0-based 0 and 2); gold size 2
    outcome = _outcome(["c1", "x", "c3"], ["c1", "c3"])
    dcg = 1 / math.log2(2) + 1 / math.log2(4)  # ranks 0 and 2
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal: ranks 0 and 1
    assert abs(NDCGAtK(3).compute(outcome).value - dcg / idcg) < 1e-9


def test_mrr():
    # first relevant at rank 2 -> 0.5
    outcome = _outcome(["x", "c1", "c3"], ["c1", "c3"])
    assert MRR().compute(outcome).value == 0.5
    # none relevant -> 0
    assert MRR().compute(_outcome(["x", "y"], ["c1"])).value == 0.0


def test_hit_rate():
    assert HitRate(2).compute(_outcome(["x", "c1"], ["c1"])).value == 1.0
    assert HitRate(2).compute(_outcome(["x", "y", "c1"], ["c1"])).value == 0.0  # c1 outside top-2


def test_recall_matches_on_doc_id():
    # gold by doc id; candidate chunk id differs but doc id matches
    gold = GoldQuery(query_id="q", query="q", relevant_doc_ids=["docA"])
    cand = _cand("some-chunk", doc_id="docA")
    outcome = QueryOutcome(gold=gold, candidates=[cand])
    assert RecallAtK(5).compute(outcome).value == 1.0


# --- operational metrics ----------------------------------------------------


def test_latency_percentiles():
    metric50, metric95 = LatencyP50(), LatencyP95()
    results = [
        metric50.compute(QueryOutcome(gold=GoldQuery(query_id=str(i), query="q"), latency_ms=v))
        for i, v in enumerate([10, 20, 30, 40, 100])
    ]
    assert metric50.aggregate(results).value == 30  # median (nearest-rank)
    results95 = [
        metric95.compute(QueryOutcome(gold=GoldQuery(query_id=str(i), query="q"), latency_ms=v))
        for i, v in enumerate([10, 20, 30, 40, 100])
    ]
    assert metric95.aggregate(results95).value == 100


def test_cost_per_query():
    metric = CostPerQuery()
    rs = [
        metric.compute(QueryOutcome(gold=GoldQuery(query_id=str(i), query="q"), cost_usd=c))
        for i, c in enumerate([0.0, 0.2, 0.4])
    ]
    assert abs(metric.aggregate(rs).value - 0.2) < 1e-9


def test_token_efficiency():
    metric = TokenEfficiency()
    out = QueryOutcome(
        gold=GoldQuery(query_id="q", query="q"), tokens_in=80, tokens_out=20, useful_tokens=20
    )
    assert metric.compute(out).value == 0.2


# --- end-to-end citation metrics (lexical judge) ----------------------------


def _gen(text: str, citations: list[Citation]) -> GenerationResult:
    from uuid import uuid4

    return GenerationResult(
        text=text, citations=citations, trace_id=uuid4(), cost=0.0, tokens_in=0, tokens_out=0
    )


def _citation(chunk_id: str, span: tuple[int, int]) -> Citation:
    return Citation(source=Source(doc_id="d", chunk_id=chunk_id), claim_span=span)


def test_citation_precision():
    text = "The sky is blue."
    gen = _gen(text, [_citation("c1", (0, len(text)))])
    cand = _cand("c1", text="the sky appears blue due to scattering")
    gold = GoldQuery(query_id="q", query="q")
    outcome = QueryOutcome(gold=gold, candidates=[cand], generation=gen)
    # claim shares "the sky blue" with evidence -> supported
    assert CitationPrecision(LexicalOverlapJudge(0.2)).compute(outcome).value == 1.0

    # citation pointing at unrelated evidence -> unsupported
    bad_cand = _cand("c1", text="completely unrelated text about finance markets")
    outcome2 = QueryOutcome(gold=gold, candidates=[bad_cand], generation=gen)
    assert CitationPrecision(LexicalOverlapJudge(0.5)).compute(outcome2).value == 0.0


def test_citation_recall():
    # two sentences, both supportable; only the first is cited -> recall 0.5
    text = "Cats purr softly. Dogs bark loudly."
    gen = _gen(text, [_citation("c1", (0, 17))])  # covers first sentence only
    cands = [
        _cand("c1", text="cats purr softly when content"),
        _cand("c2", doc_id="d2", text="dogs bark loudly at strangers"),
    ]
    gold = GoldQuery(query_id="q", query="q")
    outcome = QueryOutcome(gold=gold, candidates=cands, generation=gen)
    assert CitationRecall(LexicalOverlapJudge(0.2)).compute(outcome).value == 0.5
