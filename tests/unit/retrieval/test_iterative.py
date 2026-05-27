"""Tests for the iterative / multi-hop retriever (SPEC §7.6.7).

Two layers:

* **Unit** — judge JSON parsing, the hop loop (stop-on-done, follow-up, hop cap,
  repeat-query guard), budget gating, and dedup/re-rank.
* **Eval slice** — the SPEC §10 Phase 3 acceptance gate: on a corpus of *hard*
  multi-hop queries whose answer chunk is lexically unreachable from the original
  phrasing, iterative retrieval lifts recall@5 by >=10% over single-shot. We run
  the same dataset through both via the real :class:`EvalRunner`/``RecallAtK(5)``.

Everything is offline: the judge is scripted (no LLM) and the index is the lexical
``FakeIndex`` fixture, so the multi-hop behaviour is deterministic.
"""

from __future__ import annotations

import pytest

from common.schemas import GoldQuery, Query, RetrievalCandidate
from evaluation.datasets.loader import Dataset
from evaluation.metrics.base import QueryOutcome
from evaluation.metrics.retrieval import RecallAtK
from evaluation.runners.runner import EvalRunner
from harness.budget.tracker import BudgetTracker
from knowledge_index.retrieval.iterative import (
    HopDecision,
    IterativeRetriever,
)
from knowledge_index.retrieval.iterative.judge import LLMHopJudge, _parse_decision
from knowledge_index.retrieval.retrievers import SparseBM25Retriever
from self_improvement.registry.pipeline_config import PipelineConfig

# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class ScriptedJudge:
    """Deterministic judge: emits a registered follow-up once per original query.

    ``follow_ups`` maps an original query's raw text to the single follow-up to
    issue. The first time it sees an original it returns ``done=False`` with that
    follow-up; thereafter (or when no follow-up is registered) it returns
    ``done=True``. Mirrors what an LLM judge would do on a two-hop question.
    """

    def __init__(self, follow_ups: dict[str, str]) -> None:
        self._follow_ups = follow_ups
        self._issued: set[str] = set()

    async def judge(self, original: Query, evidence: list[RetrievalCandidate]) -> HopDecision:
        nxt = self._follow_ups.get(original.raw)
        if nxt and original.raw not in self._issued:
            self._issued.add(original.raw)
            return HopDecision(done=False, next_query=nxt, reasoning="needs hop")
        return HopDecision(done=True, next_query="", reasoning="sufficient")


class CountingRetriever:
    """Records every (query, k) it is asked for; returns a fixed candidate."""

    name = "counting"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def retrieve(self, query: Query, k: int) -> list[RetrievalCandidate]:
        self.calls.append(query.raw)
        # A unique chunk per distinct query so evidence accumulates.
        cid = f"c-{len(self.calls)}"
        return [
            RetrievalCandidate(
                chunk=_chunk(cid, query.raw),
                score=1.0,
                retriever=self.name,
                rank=1,
            )
        ]


def _chunk(chunk_id: str, text: str):
    from common.schemas import Chunk

    return Chunk(chunk_id=chunk_id, doc_id=f"doc-{chunk_id}", text=text)


# --------------------------------------------------------------------------- #
# Judge parsing
# --------------------------------------------------------------------------- #


def test_parse_decision_plain_json():
    d = _parse_decision('{"done": false, "next_query": "more", "reasoning": "gap"}')
    assert d.done is False
    assert d.follow_up == "more"


def test_parse_decision_strips_code_fence():
    raw = '```json\n{"done": true, "next_query": "", "reasoning": "ok"}\n```'
    d = _parse_decision(raw)
    assert d.done is True
    assert d.follow_up == ""


def test_parse_decision_trims_follow_up_whitespace():
    d = _parse_decision('{"done": false, "next_query": "  spaced  "}')
    assert d.follow_up == "spaced"


def test_parse_decision_rejects_non_json():
    with pytest.raises(ValueError, match="valid JSON"):
        _parse_decision("not json at all")


def test_parse_decision_rejects_json_array():
    with pytest.raises(ValueError, match="JSON object"):
        _parse_decision("[1, 2, 3]")


@pytest.mark.asyncio
async def test_llm_hop_judge_uses_injected_completer():
    captured: dict[str, str] = {}

    async def fake_complete(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"done": true, "next_query": "", "reasoning": "done"}'

    judge = LLMHopJudge(complete=fake_complete)
    q = Query(raw="who founded the lab that built it?")
    evidence = [
        RetrievalCandidate(
            chunk=_chunk("c1", "Some evidence text"), score=1.0, retriever="x", rank=1
        )
    ]
    decision = await judge.judge(q, evidence)
    assert decision.done is True
    # Prompt carries the original question and the evidence snippet.
    assert "who founded the lab" in captured["prompt"]
    assert "Some evidence text" in captured["prompt"]


# --------------------------------------------------------------------------- #
# Hop loop
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stops_when_judge_says_done():
    inner = CountingRetriever()
    judge = ScriptedJudge({})  # never asks for a follow-up
    it = IterativeRetriever(inner, judge, max_hops=4)
    await it.retrieve(Query(raw="q"), k=5)
    assert inner.calls == ["q"]  # single hop only


@pytest.mark.asyncio
async def test_follows_up_then_stops():
    inner = CountingRetriever()
    judge = ScriptedJudge({"q": "follow up query"})
    it = IterativeRetriever(inner, judge, max_hops=4)
    out = await it.retrieve(Query(raw="q"), k=5)
    assert inner.calls == ["q", "follow up query"]
    # Evidence from both hops survives dedup (distinct chunks).
    assert len(out) == 2


@pytest.mark.asyncio
async def test_respects_max_hops_cap():
    inner = CountingRetriever()

    # A judge that always wants another (distinct) hop.
    class AlwaysMore:
        def __init__(self) -> None:
            self.n = 0

        async def judge(self, original, evidence):
            self.n += 1
            return HopDecision(done=False, next_query=f"hop-{self.n}", reasoning="more")

    it = IterativeRetriever(inner, AlwaysMore(), max_hops=2)
    await it.retrieve(Query(raw="q"), k=5)
    assert len(inner.calls) == 2  # capped, never judges after the last hop


@pytest.mark.asyncio
async def test_repeat_follow_up_terminates_loop():
    inner = CountingRetriever()

    class RepeatJudge:
        async def judge(self, original, evidence):
            # Always points back at the original query text.
            return HopDecision(done=False, next_query="q", reasoning="loop")

    it = IterativeRetriever(inner, RepeatJudge(), max_hops=5)
    await it.retrieve(Query(raw="q"), k=5)
    assert inner.calls == ["q"]  # the repeat is detected, no second retrieval


def test_max_hops_must_be_positive():
    with pytest.raises(ValueError, match="max_hops"):
        IterativeRetriever(CountingRetriever(), ScriptedJudge({}), max_hops=0)


# --------------------------------------------------------------------------- #
# Budget awareness
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_budget_caps_number_of_hops():
    inner = CountingRetriever()
    judge = ScriptedJudge({"q": "follow up"})
    # Budget affords exactly one hop at the default per-hop cost.
    it = IterativeRetriever(inner, judge, max_hops=4, hop_cost_usd=0.01)
    budget = BudgetTracker(0.01)
    out = await it.retrieve(Query(raw="q"), k=5, budget=budget)
    assert inner.calls == ["q"]  # second hop unaffordable
    assert len(out) == 1
    assert budget.remaining() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_budget_allows_multiple_affordable_hops():
    inner = CountingRetriever()
    judge = ScriptedJudge({"q": "follow up"})
    it = IterativeRetriever(inner, judge, max_hops=4, hop_cost_usd=0.01)
    budget = BudgetTracker(0.05)  # room for several hops
    await it.retrieve(Query(raw="q"), k=5, budget=budget)
    assert inner.calls == ["q", "follow up"]
    assert budget.consumed == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# Dedup + re-rank
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dedup_keeps_best_score_and_reranks():
    class DupRetriever:
        name = "dup"

        def __init__(self) -> None:
            self.n = 0

        async def retrieve(self, query, k):
            self.n += 1
            # Both hops return chunk "dup", with a higher score on the 2nd hop,
            # plus a hop-unique chunk.
            return [
                RetrievalCandidate(
                    chunk=_chunk("dup", "d"), score=float(self.n), retriever=self.name, rank=1
                ),
                RetrievalCandidate(
                    chunk=_chunk(f"u{self.n}", "u"), score=0.5, retriever=self.name, rank=2
                ),
            ]

    judge = ScriptedJudge({"q": "again"})
    it = IterativeRetriever(DupRetriever(), judge, max_hops=2)
    out = await it.retrieve(Query(raw="q"), k=5)
    ids = [c.chunk.chunk_id for c in out]
    assert ids.count("dup") == 1  # collapsed
    # The surviving "dup" carries the higher (2nd-hop) score and leads the ranking.
    dup = next(c for c in out if c.chunk.chunk_id == "dup")
    assert dup.score == 2.0
    assert out[0].chunk.chunk_id == "dup"
    assert [c.rank for c in out] == list(range(1, len(out) + 1))


# --------------------------------------------------------------------------- #
# Eval slice: SPEC §10 Phase 3 acceptance — recall@5 +>=10% vs single-shot
# --------------------------------------------------------------------------- #


def _hard_corpus(make_chunk_fn):
    """Two-hop corpus: each answer chunk is reachable only via a follow-up query.

    Per hard query the original phrasing matches a *bridge* chunk (naming an
    intermediate entity) and four *distractors* — five positive-scoring chunks
    that fill a top-5 single-shot result. The *answer* chunk shares no terms with
    the original query (score 0), so single-shot never surfaces it. A follow-up
    query keyed on the bridge entity scores the answer chunk highest, so the
    multi-hop loop recovers it. (``FakeIndex.search_sparse`` scores by summed
    substring term counts, so vocabulary is kept deliberately non-overlapping.)
    """
    return [
        # Q1: original terms {billing, service}; bridge entity "Photon framework".
        make_chunk_fn("a1-bridge", "Billing service runs on Photon framework platform."),
        make_chunk_fn("a1-d1", "Billing service overview."),
        make_chunk_fn("a1-d2", "Billing service pricing."),
        make_chunk_fn("a1-d3", "Billing service uptime."),
        make_chunk_fn("a1-d4", "Billing service roadmap."),
        make_chunk_fn("a1-answer", "Photon framework implementation uses Rust."),
        # Q2: original terms {search, ranker}; bridge entity "Atlas crew".
        make_chunk_fn("a2-bridge", "Search ranker owned by Atlas crew."),
        make_chunk_fn("a2-d1", "Search ranker benchmarks."),
        make_chunk_fn("a2-d2", "Search ranker latency."),
        make_chunk_fn("a2-d3", "Search ranker config."),
        make_chunk_fn("a2-d4", "Search ranker rollout."),
        make_chunk_fn("a2-answer", "Atlas crew director: Mara Quinn."),
        # Q3: original terms {returns, pipeline}; bridge entity "depot Hub-7".
        make_chunk_fn("a3-bridge", "Returns pipeline routes through depot Hub-7."),
        make_chunk_fn("a3-d1", "Returns pipeline SLA."),
        make_chunk_fn("a3-d2", "Returns pipeline refunds."),
        make_chunk_fn("a3-d3", "Returns pipeline metrics."),
        make_chunk_fn("a3-d4", "Returns pipeline owners."),
        make_chunk_fn("a3-answer", "Depot Hub-7 location: Reno Nevada address."),
    ]


def _hard_dataset() -> Dataset:
    return Dataset(
        name="iterative-hard",
        queries=[
            GoldQuery(
                query_id="q1",
                query="what language powers the billing service",
                relevant_chunk_ids=["a1-answer"],
                difficulty="hard",
            ),
            GoldQuery(
                query_id="q2",
                query="who leads the team behind the search ranker",
                relevant_chunk_ids=["a2-answer"],
                difficulty="hard",
            ),
            GoldQuery(
                query_id="q3",
                query="where is the warehouse for the returns pipeline",
                relevant_chunk_ids=["a3-answer"],
                difficulty="hard",
            ),
        ],
    )


# Follow-up queries the judge emits per original — keyed on the bridge entity so
# the answer chunk becomes lexically reachable on hop 2.
_HARD_FOLLOW_UPS = {
    "what language powers the billing service": "Photon framework implementation",
    "who leads the team behind the search ranker": "Atlas crew director",
    "where is the warehouse for the returns pipeline": "depot Hub-7 location address",
}


class _RetrieverRunner:
    """Adapts a bare :class:`Retriever` into an eval ``PipelineRunner``."""

    def __init__(self, retriever) -> None:
        self._retriever = retriever

    async def run_query(self, query: GoldQuery, *, k: int) -> QueryOutcome:
        candidates = await self._retriever.retrieve(Query(raw=query.query), k)
        return QueryOutcome(gold=query, candidates=candidates)


@pytest.mark.asyncio
async def test_eval_slice_iterative_beats_single_shot_recall_at_5(fake_index_cls, make_chunk_fn):
    index = fake_index_cls(_hard_corpus(make_chunk_fn))
    single = SparseBM25Retriever(index)
    iterative = IterativeRetriever(single, ScriptedJudge(_HARD_FOLLOW_UPS), max_hops=2)

    dataset = _hard_dataset()
    runner_single = EvalRunner(_RetrieverRunner(single), k=5)
    runner_iter = EvalRunner(_RetrieverRunner(iterative), k=5)
    config = PipelineConfig()
    metrics = [RecallAtK(5)]

    report_single = await runner_single.run(config, dataset, metrics)
    report_iter = await runner_iter.run(config, dataset, metrics)

    base = report_single.recall_at(5)
    improved = report_iter.recall_at(5)

    # Single-shot genuinely misses the answer chunks (else the slice isn't testing
    # multi-hop); iterative recovers every one.
    assert base < improved
    assert improved >= 0.99  # iterative recalls every hard answer
    # SPEC §10 Phase 3 acceptance: recall@5 improves by >=10% on hard queries.
    # Stated as an absolute gain (unambiguous when the single-shot baseline is 0).
    assert improved - base >= 0.10
    assert improved >= base * 1.10
