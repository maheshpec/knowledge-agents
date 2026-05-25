"""LLM query rewriter (SPEC §7.6.2) — the one query op implemented in Phase 1.

Reformulates a conversational/underspecified query into a cleaner search query
(resolves pronouns, drops filler, surfaces key terms). The rewrite is *appended*
to ``query.rewrites`` rather than replacing ``raw`` so downstream stages can still
see the original — the dense retriever prefers the rewrite, BM25 keeps the raw.
"""

from __future__ import annotations

from common.schemas import Query
from harness.observability.tracing import traced
from knowledge_index.retrieval.query_ops.base import CompleteFn, default_completer

REWRITE_PROMPT = """Rewrite the following search query to maximize retrieval quality.
Resolve references, remove conversational filler, and keep the key terms.
Answer with only the rewritten query and nothing else.

Query: {query}"""


class Rewriter:
    """Rewrite the query for retrieval via an LLM."""

    name = "rewriter"

    def __init__(self, complete: CompleteFn | None = None) -> None:
        self._complete = complete or default_completer()

    @traced(span_name="retrieval.query_ops.rewriter")
    async def transform(self, query: Query) -> Query:
        rewritten = (await self._complete(REWRITE_PROMPT.format(query=query.raw))).strip()
        if not rewritten or rewritten == query.raw or rewritten in query.rewrites:
            return query
        return query.model_copy(update={"rewrites": [*query.rewrites, rewritten]})


__all__ = ["REWRITE_PROMPT", "Rewriter"]
