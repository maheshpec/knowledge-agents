"""Tests for the query router (SPEC §7.6.1, Phase 2G).

Covers the RouteDecision schema, the LLMRouter (parsing + aggressive caching),
a hand-labeled 20-query fixture mapping queries → expected strategies, and the
RouterPipeline (strategy selection + graph/iterative fallback to hybrid).
All LLM calls use an injected fake completer, so these run fully offline.
"""

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from common.schemas import Query, RetrievalResult
from knowledge_index.retrieval.routers import (
    LLMRouter,
    RouteDecision,
    RouterPipeline,
)

# Hand-labeled fixture set: query → (strategy, intent, complexity). 20 queries
# spanning all four strategies and intents (SPEC §7.6.1 acceptance).
LABELED: list[tuple[str, str, str, str]] = [
    ("What is the capital of France?", "naive", "lookup", "low"),
    ("Define idempotency.", "naive", "lookup", "low"),
    ("What port does the service listen on?", "naive", "lookup", "low"),
    ("When was the company founded?", "naive", "lookup", "low"),
    ("What is the default timeout value?", "naive", "lookup", "low"),
    ("Summarize the incident response policy.", "hybrid", "synthesis", "med"),
    ("Give an overview of our caching strategy.", "hybrid", "synthesis", "med"),
    ("Explain how the ingestion pipeline works.", "hybrid", "synthesis", "med"),
    ("What are the key themes in the design docs?", "hybrid", "synthesis", "med"),
    ("Describe the onboarding process.", "hybrid", "synthesis", "med"),
    ("Compare the 2023 and 2024 revenue guidance.", "iterative", "comparison", "high"),
    ("Contrast REST and gRPC for our use case.", "iterative", "comparison", "high"),
    ("How do approaches A and B differ in latency?", "iterative", "comparison", "high"),
    ("Which is cheaper: option X or option Y, and why?", "iterative", "comparison", "high"),
    ("Synthesize the tradeoffs across all three proposals.", "iterative", "synthesis", "high"),
    ("How is the auth service related to billing?", "graph", "relational", "med"),
    ("What depends on the payments module?", "graph", "relational", "med"),
    ("Trace the call path from API to database.", "graph", "relational", "high"),
    ("Which teams own services connected to checkout?", "graph", "relational", "med"),
    ("Map the relationships between the data models.", "graph", "relational", "med"),
]


def _labeled_completer():
    """A CompleteFn that returns the labeled decision for whichever query appears."""
    table = {
        q: {"strategy": s, "intent": i, "expected_complexity": c, "filters": {}}
        for q, s, i, c in LABELED
    }

    async def _complete(prompt: str) -> str:
        # The real query is the trailing "Query: ..." line (few-shot examples use
        # the same prefix earlier in the prompt, so match the last occurrence).
        query = prompt.rsplit("Query: ", 1)[-1].strip()
        if query not in table:
            raise AssertionError(f"unexpected query in prompt: {query!r}")
        return json.dumps(table[query])

    return _complete


def _fixed_completer(decision: dict, counter: list[int]):
    async def _complete(prompt: str) -> str:
        counter[0] += 1
        return json.dumps(decision)

    return _complete


def test_route_decision_defaults():
    d = RouteDecision()
    assert d.strategy == "hybrid"
    assert d.intent == "lookup"
    assert d.expected_complexity == "low"
    assert d.filters == {}


def test_route_decision_rejects_bad_strategy():
    with pytest.raises(ValidationError):
        RouteDecision(strategy="quantum")  # not in the Literal


async def test_router_classifies_labeled_fixture_set():
    router = LLMRouter(_labeled_completer(), corpus_vocab=["billing", "auth", "payments"])
    correct = 0
    for q, strategy, intent, complexity in LABELED:
        decision = await router.route(Query(raw=q))
        assert decision.strategy == strategy, q
        assert decision.intent == intent, q
        assert decision.expected_complexity == complexity, q
        correct += 1
    assert correct == 20


async def test_router_caches_per_query_hash():
    counter = [0]
    decision = {"strategy": "hybrid", "intent": "synthesis", "expected_complexity": "med"}
    router = LLMRouter(_fixed_completer(decision, counter))
    q = Query(raw="repeat me")
    first = await router.route(q)
    second = await router.route(Query(raw="repeat me"))
    assert first.strategy == second.strategy
    assert counter[0] == 1  # LLM invoked once; second served from cache


async def test_router_rejects_invalid_json():
    async def bad(prompt: str) -> str:
        return "definitely not json"

    with pytest.raises(ValueError):
        await LLMRouter(bad).route(Query(raw="q"))


# ---- RouterPipeline ---------------------------------------------------------


class _StubPipeline:
    """A SupportsRetrieve that records the query it was asked to retrieve."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[Query] = []

    async def retrieve(self, query: Query, k: int) -> RetrievalResult:
        self.calls.append(query)
        return RetrievalResult(candidates=[], query=query, trace_id=uuid4())


class _StubRouter:
    name = "stub"

    def __init__(self, decision: RouteDecision):
        self._decision = decision

    async def route(self, query: Query) -> RouteDecision:
        return self._decision


async def test_pipeline_routes_to_selected_variant():
    naive = _StubPipeline("naive")
    hybrid = _StubPipeline("hybrid")
    router = _StubRouter(RouteDecision(strategy="naive", intent="lookup"))
    pipe = RouterPipeline(router, hybrid, variants={"naive": naive, "hybrid": hybrid})
    await pipe.retrieve(Query(raw="q"), k=5)
    assert len(naive.calls) == 1
    assert len(hybrid.calls) == 0


async def test_pipeline_falls_back_to_hybrid_for_graph_and_iterative():
    hybrid = _StubPipeline("hybrid")
    for strategy in ("graph", "iterative"):
        router = _StubRouter(RouteDecision(strategy=strategy, intent="relational"))
        pipe = RouterPipeline(router, hybrid)
        await pipe.retrieve(Query(raw="q"), k=5)
    assert len(hybrid.calls) == 2  # both fell back to the hybrid pipeline


async def test_pipeline_stamps_intent_and_filters_onto_query():
    hybrid = _StubPipeline("hybrid")
    decision = RouteDecision(strategy="hybrid", intent="comparison", filters={"team": "core"})
    pipe = RouterPipeline(_StubRouter(decision), hybrid)
    await pipe.retrieve(Query(raw="q"), k=5)
    routed = hybrid.calls[0]
    assert routed.intent == "comparison"
    assert routed.filters == {"team": "core"}


async def test_pipeline_query_filters_take_precedence():
    hybrid = _StubPipeline("hybrid")
    decision = RouteDecision(strategy="hybrid", filters={"team": "core", "year": "2023"})
    pipe = RouterPipeline(_StubRouter(decision), hybrid)
    await pipe.retrieve(Query(raw="q", filters={"team": "ml"}), k=5)
    # caller-supplied filters win over the router's inferred ones
    assert hybrid.calls[0].filters == {"team": "ml", "year": "2023"}
