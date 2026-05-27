"""Retriever protocol, index/embedder contracts, and the parallel harness (SPEC §7.6.3).

Convoy B owns the concrete ``QdrantIndex`` and ``Embedder`` implementations; they
are not present on this branch. Rather than import them, retrievers depend on the
*structural* protocols below — any object exposing the right ``async`` methods
satisfies the contract, so B's classes drop in unchanged at integration time.

ACL enforcement is pushed down into the index (SPEC §7.5): retrievers attach the
caller's principals to the filter dict; the index applies it as a payload filter
so mismatched principals return zero hits *before* anything leaves the store.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from common.schemas import Query, RetrievalCandidate
from harness.observability.tracing import traced

# Filter key under which the caller's principals are passed to the index. The
# index treats a candidate as visible iff its ``acl`` payload intersects this set
# (empty principals + empty chunk acl => public, visible to all).
ACL_FILTER_KEY = "acl"


@runtime_checkable
class SupportsSearch(Protocol):
    """The slice of Convoy B's ``Index`` (SPEC §7.5) that retrievers consume."""

    async def search_dense(
        self, vec: list[float], k: int, filters: dict[str, Any]
    ) -> list[RetrievalCandidate]: ...

    async def search_sparse(
        self, query: str, k: int, filters: dict[str, Any]
    ) -> list[RetrievalCandidate]: ...


@runtime_checkable
class SupportsEmbedQuery(Protocol):
    """The slice of Convoy B's ``Embedder`` (SPEC §7.4) that retrievers consume."""

    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Retriever(Protocol):
    """A single retrieval strategy (SPEC §7.6.3)."""

    name: str

    async def retrieve(self, query: Query, k: int) -> list[RetrievalCandidate]: ...


def build_search_filters(query: Query) -> dict[str, Any]:
    """Compose the index filter dict for ``query``, injecting ACL principals.

    The caller's principals (from the authenticated session) always go in under
    :data:`ACL_FILTER_KEY` so the index can enforce access control uniformly.

    Security: ``query.filters`` is NOT trusted to set access control. Any
    ``acl`` / ``user_principals`` keys carried on ``query.filters`` are stripped
    before merging, so a caller cannot widen its own principal set by smuggling
    an ACL key through query-level filters (a privilege-escalation surface —
    these filters can originate from query-influenced routing; see SPEC §11 #6
    and §13). Only the authenticated ``query.user_principals`` decides visibility.
    """
    filters: dict[str, Any] = {
        k: v for k, v in query.filters.items() if k not in (ACL_FILTER_KEY, "user_principals")
    }
    filters[ACL_FILTER_KEY] = list(query.user_principals)
    return filters


@traced(span_name="retrieval.retrievers.gather")
async def gather_retrievers(
    retrievers: list[Retriever], query: Query, k: int
) -> list[list[RetrievalCandidate]]:
    """Run every retriever concurrently and return one candidate list per retriever.

    Order of the returned lists matches ``retrievers`` so a fuser can apply
    per-retriever weights positionally. Exceptions propagate (fail-fast); the
    pipeline decides whether a partial failure is recoverable.
    """
    if not retrievers:
        return []
    return await asyncio.gather(*(r.retrieve(query, k) for r in retrievers))


__all__ = [
    "ACL_FILTER_KEY",
    "SupportsSearch",
    "SupportsEmbedQuery",
    "Retriever",
    "build_search_filters",
    "gather_retrievers",
]
