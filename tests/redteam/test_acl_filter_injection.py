"""Red-team: ACL privilege escalation via injected query-level filters.

Attacker goal: widen their own access by smuggling an ``acl`` / ``user_principals``
key into ``query.filters`` (these filters can originate from query-influenced
routing, so they are attacker-reachable). The authenticated principal set must
remain authoritative — query filters can scope metadata but NEVER grant access.

Defense under test: ``build_search_filters`` strips ACL keys from query.filters
before they reach the index. Verified both in isolation and end-to-end against
the real in-memory QdrantIndex.
"""

from __future__ import annotations

from common.schemas import Query
from knowledge_index.retrieval.retrievers.base import ACL_FILTER_KEY, build_search_filters
from knowledge_index.retrieval.retrievers.dense import DenseRetriever

PRIVATE_IDS = {"team-a-1", "team-b-pii", "exec-only"}


def test_acl_key_in_query_filters_cannot_override_principals(principals):
    q = Query(
        raw="x",
        user_principals=[principals["intruder"]],
        filters={"acl": ["role:exec", "team-b"]},
    )
    filters = build_search_filters(q)
    # Only the authenticated principal survives; injected acl is dropped.
    assert filters[ACL_FILTER_KEY] == [principals["intruder"]]


def test_user_principals_key_in_query_filters_cannot_augment(principals):
    q = Query(
        raw="x",
        user_principals=[principals["intruder"]],
        filters={"user_principals": ["role:exec"]},
    )
    filters = build_search_filters(q)
    assert filters[ACL_FILTER_KEY] == [principals["intruder"]]
    # The escalation key must not leak through under either name.
    assert "user_principals" not in filters
    assert "role:exec" not in filters.get(ACL_FILTER_KEY, [])


def test_benign_query_filters_are_preserved():
    # Hardening must not break legitimate metadata scoping.
    q = Query(raw="x", user_principals=["team-a"], filters={"lang": "en", "doc_id": "d1"})
    filters = build_search_filters(q)
    assert filters[ACL_FILTER_KEY] == ["team-a"]
    assert filters["lang"] == "en"
    assert filters["doc_id"] == "d1"


async def test_injection_does_not_leak_through_real_index(index, embedder, principals):
    # End-to-end: intruder tries to read exec/team-b data by injecting acl filters.
    q = Query(
        raw="rotate the production credentials",
        user_principals=[principals["intruder"]],
        filters={"acl": ["role:exec", "team-b"]},
    )
    hits = await DenseRetriever(index, embedder).retrieve(q, k=20)
    ids = {c.chunk.chunk_id for c in hits}
    assert ids == {"public-1", "public-2"}
    assert not (ids & PRIVATE_IDS)
