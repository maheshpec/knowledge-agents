"""Deduplication post-processor (SPEC §7.6.6).

Drops repeated chunks, keeping the first (highest-ranked) occurrence. Dedupes on
``chunk_id`` and, additionally, on a normalized hash of the text so that distinct
chunk ids carrying identical content (e.g. boilerplate duplicated across docs)
collapse to one. Ranks are renumbered over the survivors.
"""

from __future__ import annotations

import hashlib
import re

from common.schemas import Query, RetrievalCandidate
from harness.observability.tracing import traced

_WS = re.compile(r"\s+")


def _text_hash(text: str) -> str:
    normalized = _WS.sub(" ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class DeduplicatorPostProcessor:
    """Remove duplicate chunks by id and by normalized-text hash."""

    name = "deduplicator"

    @traced(span_name="retrieval.post.dedup")
    async def process(
        self, query: Query, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        out: list[RetrievalCandidate] = []

        for candidate in candidates:
            chunk_id = candidate.chunk.chunk_id
            text_key = _text_hash(candidate.chunk.text)
            if chunk_id in seen_ids or text_key in seen_text:
                continue
            seen_ids.add(chunk_id)
            seen_text.add(text_key)
            out.append(candidate)

        return [c.model_copy(update={"rank": i}) for i, c in enumerate(out, start=1)]


__all__ = ["DeduplicatorPostProcessor"]
