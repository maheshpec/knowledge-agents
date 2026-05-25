"""Parent expansion post-processor (SPEC §7.6.6).

Small chunks retrieve precisely but read poorly in isolation. This swaps each
matched chunk for its parent (the larger enclosing chunk) so the generator sees
full context. When several children of the same parent are retrieved, the parent
appears once, carrying the best child's score.

Parents are fetched via an injected ``fetch_parent`` callable (an index lookup at
integration time); a candidate whose parent is missing or has no ``parent_id`` is
left untouched.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from common.schemas import Chunk, Query, RetrievalCandidate
from harness.observability.tracing import traced

FetchParentFn = Callable[[str], Awaitable[Chunk | None]]


class ParentExpander:
    """Replace matched chunks with their parent chunks, deduping shared parents."""

    name = "parent_expander"

    def __init__(self, fetch_parent: FetchParentFn) -> None:
        self._fetch_parent = fetch_parent

    @traced(span_name="retrieval.post.parent_expander")
    async def process(
        self, query: Query, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        out: list[RetrievalCandidate] = []
        seen_parents: dict[str, int] = {}  # parent chunk_id -> index in `out`

        for candidate in candidates:
            parent_id = candidate.chunk.parent_id
            parent = await self._fetch_parent(parent_id) if parent_id else None

            if parent is None:
                out.append(candidate)
                continue

            if parent.chunk_id in seen_parents:
                # Keep the higher-scoring child's score on the single parent entry.
                existing = out[seen_parents[parent.chunk_id]]
                if candidate.score > existing.score:
                    out[seen_parents[parent.chunk_id]] = existing.model_copy(
                        update={"score": candidate.score}
                    )
                continue

            seen_parents[parent.chunk_id] = len(out)
            out.append(candidate.model_copy(update={"chunk": parent}))

        return [c.model_copy(update={"rank": i}) for i, c in enumerate(out, start=1)]


__all__ = ["FetchParentFn", "ParentExpander"]
