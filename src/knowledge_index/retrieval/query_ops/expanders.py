"""LLM-backed query expanders: HyDE, decomposition, step-back (SPEC §7.6.2).

These complete the Phase 1 :class:`Rewriter` with the rest of the query-op zoo.
Each takes an injected ``complete`` callable (``str -> str``) so it runs offline
under test; the default is :func:`default_completer`, built lazily on first use.

Each op writes into a *different* field of :class:`Query` so they compose without
clobbering one another (``[Rewriter(), HyDEExpander(), Decomposer()]``):

- :class:`HyDEExpander`  → ``query.hyde``        (hypothetical answer documents)
- :class:`Decomposer`    → ``query.sub_queries`` (atomic sub-questions)
- :class:`Stepback`      → ``query.rewrites``    (a broader reformulation)

The transform is additive and idempotent-ish: empty / duplicate / echo outputs are
dropped so a misbehaving model can never poison the query.
"""

from __future__ import annotations

import re

from common.schemas import Query
from harness.observability.tracing import traced
from knowledge_index.retrieval.query_ops.base import CompleteFn, default_completer

HYDE_PROMPT = """Write a short, factual passage that would directly answer the \
following question, as if it were an excerpt from an authoritative document. \
Do not hedge or say you are unsure — write the ideal answer passage. Keep it to \
one paragraph. Answer with only the passage.

Question: {query}"""

DECOMPOSE_PROMPT = """Break the following question into the minimal set of \
independent sub-questions that must each be answered to answer it fully. \
Output one sub-question per line, with no numbering or bullets. If the question \
is already atomic, output it unchanged on a single line.

Question: {query}"""

STEPBACK_PROMPT = """Given the following specific question, generate a single \
broader, more general 'step-back' question whose answer would provide useful \
background for answering the original. Answer with only the step-back question.

Question: {query}"""

# Strip a leading list marker ("1.", "1)", "-", "*", "•") from a decomposed line.
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")


class HyDEExpander:
    """Hypothetical Document Embeddings: generate a fake answer, embed it (SPEC §7.6.2).

    The hypothetical passage is appended to ``query.hyde``; the dense retriever
    embeds it instead of (or alongside) the raw query, which often lands closer to
    real answer chunks in vector space than the question does.
    """

    name = "hyde"

    def __init__(self, complete: CompleteFn | None = None) -> None:
        self._complete = complete or default_completer()

    @traced(span_name="retrieval.query_ops.hyde")
    async def transform(self, query: Query) -> Query:
        passage = (await self._complete(HYDE_PROMPT.format(query=query.raw))).strip()
        if not passage or passage in query.hyde:
            return query
        return query.model_copy(update={"hyde": [*query.hyde, passage]})


class Decomposer:
    """Break a multi-part query into atomic sub-queries (SPEC §7.6.2).

    Sub-questions are appended to ``query.sub_queries`` for fan-out retrieval. A
    decomposition that yields a single line equal to the original is treated as a
    no-op (the query was already atomic).
    """

    name = "decompose"

    def __init__(self, complete: CompleteFn | None = None) -> None:
        self._complete = complete or default_completer()

    @traced(span_name="retrieval.query_ops.decompose")
    async def transform(self, query: Query) -> Query:
        raw = await self._complete(DECOMPOSE_PROMPT.format(query=query.raw))
        subs: list[str] = []
        for line in raw.splitlines():
            cleaned = _LIST_MARKER.sub("", line).strip()
            if cleaned and cleaned not in subs:
                subs.append(cleaned)
        # No useful split: single sub-question identical to the original query.
        if len(subs) <= 1 and (not subs or subs[0] == query.raw):
            return query
        merged = [*query.sub_queries]
        merged.extend(s for s in subs if s not in merged)
        return query.model_copy(update={"sub_queries": merged})


class Stepback:
    """Generate a broader 'step-back' reformulation of the query (SPEC §7.6.2).

    The broader question is appended to ``query.rewrites`` so retrievers can run it
    as an alternate phrasing — useful when the specific query is too narrow to
    match the relevant background passages.
    """

    name = "stepback"

    def __init__(self, complete: CompleteFn | None = None) -> None:
        self._complete = complete or default_completer()

    @traced(span_name="retrieval.query_ops.stepback")
    async def transform(self, query: Query) -> Query:
        broader = (await self._complete(STEPBACK_PROMPT.format(query=query.raw))).strip()
        if not broader or broader == query.raw or broader in query.rewrites:
            return query
        return query.model_copy(update={"rewrites": [*query.rewrites, broader]})


__all__ = [
    "HYDE_PROMPT",
    "DECOMPOSE_PROMPT",
    "STEPBACK_PROMPT",
    "HyDEExpander",
    "Decomposer",
    "Stepback",
]
