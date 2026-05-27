"""LLM-backed query router (SPEC §7.6.1).

``LLMRouter`` prompts a cheap, fast model (Haiku 4.5 by default) with a few-shot
prompt and a sample of the corpus topic vocabulary, and parses a
:class:`RouteDecision` out of the JSON reply. Decisions are cached aggressively
per ``query_hash`` via the Phase 1A :class:`~harness.cache.retrieval_cache.RetrievalCache`
so repeated/near-repeated queries skip the LLM call entirely.

The LLM call is an injected ``CompleteFn`` so the router runs offline under test.
"""

from __future__ import annotations

import json

from common.schemas import Query
from harness.cache.retrieval_cache import RetrievalCache, retrieval_cache_key
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from knowledge_index.retrieval.query_ops.base import CompleteFn, default_completer
from knowledge_index.retrieval.routers.base import RouteDecision

_log = get_logger("knowledge_index.retrieval.routers.llm")

ROUTER_PROMPT = """You are a retrieval router. Classify the query and pick the best \
retrieval strategy.

Return ONLY a JSON object (no prose, no code fences):
  {{"strategy": "naive|hybrid|graph|iterative",
    "intent": "lookup|synthesis|comparison|relational",
    "expected_complexity": "low|med|high",
    "filters": {{}}}}

Guidance:
- naive: a single fact lookup answerable by one passage.
- hybrid: most queries — keyword + semantic retrieval.
- graph: relational questions spanning linked entities ("how does X connect to Y").
- iterative: multi-hop synthesis/comparison needing several retrieval rounds.
- intent: lookup (one fact), synthesis (combine sources), comparison (contrast),
  relational (entity relationships).

Examples:
Query: "What is the capital of France?"
{{"strategy": "naive", "intent": "lookup", "expected_complexity": "low", "filters": {{}}}}
Query: "Compare the 2023 and 2024 revenue guidance."
{{"strategy": "iterative", "intent": "comparison", "expected_complexity": "high", "filters": {{}}}}
Query: "How is the auth service related to the billing module?"
{{"strategy": "graph", "intent": "relational", "expected_complexity": "med", "filters": {{}}}}
Query: "Summarize our incident response policy."
{{"strategy": "hybrid", "intent": "synthesis", "expected_complexity": "med", "filters": {{}}}}

Corpus topics (sample): {vocab}

Query: {query}"""

DEFAULT_ROUTER_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_INDEX_VERSION = "v0"
_MAX_VOCAB_TERMS = 40


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_decision(raw: str) -> RouteDecision:
    """Parse the LLM reply into a RouteDecision, tolerating fences/extra keys."""
    payload = _strip_code_fence(raw)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"router LLM did not return valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("router LLM must return a JSON object")
    # Pydantic validates the Literal fields and ignores unknown keys by default.
    return RouteDecision.model_validate(data)


class LLMRouter:
    """Route queries to a retrieval strategy via a few-shot LLM classifier."""

    name = "llm_router"

    def __init__(
        self,
        complete: CompleteFn | None = None,
        *,
        corpus_vocab: list[str] | None = None,
        cache: RetrievalCache[RouteDecision] | None = None,
        index_version: str = DEFAULT_INDEX_VERSION,
    ) -> None:
        self._complete = complete or default_completer(DEFAULT_ROUTER_MODEL)
        self._vocab = list(corpus_vocab or [])
        self._cache = cache if cache is not None else RetrievalCache[RouteDecision]()
        self._index_version = index_version

    def _vocab_sample(self) -> str:
        return ", ".join(self._vocab[:_MAX_VOCAB_TERMS]) if self._vocab else "(none provided)"

    @traced(span_name="retrieval.routers.llm")
    async def route(self, query: Query) -> RouteDecision:
        key = retrieval_cache_key(query.raw, self._index_version, query.filters)
        cached = self._cache.get(key)
        if cached is not None:
            _log.info("router.cache_hit", strategy=cached.strategy)
            return cached

        prompt = ROUTER_PROMPT.format(vocab=self._vocab_sample(), query=query.raw)
        decision = _parse_decision(await self._complete(prompt))
        # Carry over any caller-supplied filters the LLM didn't surface.
        if query.filters and not decision.filters:
            decision = decision.model_copy(update={"filters": dict(query.filters)})
        self._cache.put(key, decision)
        _log.info(
            "router.decided",
            strategy=decision.strategy,
            intent=decision.intent,
            complexity=decision.expected_complexity,
        )
        return decision


__all__ = ["LLMRouter", "ROUTER_PROMPT", "DEFAULT_ROUTER_MODEL", "DEFAULT_INDEX_VERSION"]
