"""Tests for the DCI heuristic router (SPEC §15.2, Phase 5B).

Golden table mapping query → expected strategy, plus exercises for the tunable
weights (so the Phase 4 evolutionary loop has a regression net when it tweaks
thresholds).
"""

from __future__ import annotations

import pytest

from common.schemas import Query
from knowledge_index.retrieval.routers import HeuristicRouter
from knowledge_index.retrieval.routers.base import DCI_STRATEGIES

# (query, expected strategy). Hand-labeled per the §15.2 routing heuristics:
# quoted phrases / identifier-like tokens / code-style queries → dci; multi-hop
# named-entity bridges → dci_then_vector; paraphrastic / large-corpus → hybrid.
GOLDEN: list[tuple[str, str]] = [
    # --- pure DCI: quotes, identifiers, code keywords ----------------------
    ('Find the phrase "transparent failover" in the docs', "dci"),
    ('Where is "MAX_RETRY_COUNT" defined?', "dci"),
    ("Show me the `parse_config` function", "dci"),
    ("Where is foo_bar_baz used?", "dci"),
    ("Find class UserAuthHandler", "dci"),
    ("def handle_request implementations", "dci"),
    ("imports of payments.api in the codebase", "dci"),
    ("look in services/auth/handler.py", "dci"),
    # --- chained bridge: ≥2 proper nouns + relational word -----------------
    ("How does AuthService connect to BillingModule?", "dci_then_vector"),
    ('Compare "ServiceA" with ServiceB latency budget', "dci_then_vector"),
    # --- vector hybrid: paraphrastic synthesis, no lexical anchors ---------
    (
        "Explain how our caching strategy reduces tail latency across the "
        "ingestion pipeline and downstream services",
        "hybrid",
    ),
    (
        "Summarize the principles behind our team's approach to "
        "incident response and operational excellence",
        "hybrid",
    ),
    ("What is the capital of France?", "hybrid"),
    ("Describe our onboarding philosophy", "hybrid"),
]


@pytest.mark.parametrize("query, expected", GOLDEN)
def test_golden_table(query: str, expected: str) -> None:
    router = HeuristicRouter()
    decision = router.decide(query)
    assert decision.strategy == expected, (
        f"query={query!r} got strategy={decision.strategy!r} (reasoning={decision.reasoning})"
    )


async def test_route_protocol_matches_sync_decide() -> None:
    router = HeuristicRouter()
    sync = router.decide('What does "MAX_RETRY_COUNT" mean?')
    async_ = await router.route(Query(raw='What does "MAX_RETRY_COUNT" mean?'))
    assert sync.strategy == async_.strategy == "dci"


def test_lowering_quote_weight_flips_dci_to_hybrid() -> None:
    # Phase 4 should be able to tune the heuristic — at quote_weight=0 a quoted
    # query without other DCI signals must fall back to vector hybrid.
    q = 'find "alpha beta" please'
    assert HeuristicRouter().decide(q).strategy == "dci"
    assert HeuristicRouter(quote_weight=0.0).decide(q).strategy == "hybrid"


def test_chained_order_follows_dominant_signal() -> None:
    # When both DCI and vector signals are present alongside a bridge, the
    # order should put the stronger signal first.
    dci_first = HeuristicRouter().decide(
        'How does "AuthService" connect to BillingModule and PaymentsModule?'
    )
    assert dci_first.strategy == "dci_then_vector"
    # Vector signal dominates: long paraphrastic query, only the bridge has
    # any DCI weight from proper nouns. We synthesise the case with a router
    # whose long_query weight outranks identifier weight.
    decision = HeuristicRouter(
        identifier_weight=0.1,
        long_query_vector_weight=5.0,
    ).decide(
        "Could you walk me through how the AuthService relates to the "
        "BillingModule when we are reasoning about retries and timeouts in "
        "the global rollout plan and connect that to operational excellence"
    )
    assert decision.strategy == "vector_then_dci"


def test_all_emitted_strategies_are_known() -> None:
    # Guards against the router emitting a strategy the Strategy Literal
    # doesn't accept; if this ever fires, base.py and heuristic.py have drifted.
    router = HeuristicRouter()
    allowed = {"naive", "hybrid", "graph", "iterative"} | DCI_STRATEGIES
    for query, _ in GOLDEN:
        assert router.decide(query).strategy in allowed
