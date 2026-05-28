"""Heuristic query router for DCI vs vector strategies (SPEC §15.2).

The Phase 5B router that picks between the DCI strategy variants
(``dci`` / ``dci_then_vector`` / ``vector_then_dci``) and the existing vector
``hybrid`` strategy based on lightweight lexical signals — no LLM call.

Why heuristic, not LLM? Sun et al. (2026, "Beyond Semantic Similarity") and
LlamaIndex's 2026 fs-explorer benchmarks show DCI tools beat vector RAG on
exact-lexical and code-like queries, while vector hybrid still wins on
paraphrastic queries. The discriminating signals (quotes, identifiers, code
syntax) are cheap to extract, so a cached heuristic gives the routing decision
for free; an LLM call here would dominate the budget on short corpora.

Thresholds live in :attr:`HeuristicRouter` ctor params and are mirrored in
``configs/components.yaml`` (SPEC §8.1) so the Phase 4 evolutionary loop can
tune them per workload without code edits.
"""

from __future__ import annotations

import re

from common.schemas import Query
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from knowledge_index.retrieval.routers.base import (
    Complexity,
    Intent,
    RouteDecision,
    Strategy,
)

_log = get_logger("knowledge_index.retrieval.routers.heuristic")

# A quoted span like "foo bar" or 'foo bar' (≥ 2 chars between quotes). Used as
# the strongest DCI signal: users quote a phrase when they want an exact match.
_QUOTED_PHRASE = re.compile(r'"[^"]{2,}"|\'[^\']{2,}\'')

# camelCase / PascalCase / snake_case / dotted identifiers, plus things that
# look like file paths (anything with a path separator or a CamelCase token).
_IDENTIFIER = re.compile(
    r"\b(?:[a-z]+(?:_[a-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b"
)
_PATH_LIKE = re.compile(r"\b\S*[\\/]\S+\b|\S+\.(?:py|js|ts|go|java|rs|cpp|h|md)\b")
# Dotted module/path identifiers (``payments.api``, ``foo.bar.baz``). Common in
# code-style queries — users often paste a Python module path verbatim.
_DOTTED_NAME = re.compile(r"\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_.]+\b")

# Backticked spans (`foo`) and common code keywords. Backticks are an explicit
# code-fence signal — a user copying a name from a codebase typically wraps it.
_BACKTICKED = re.compile(r"`[^`]+`")
_CODE_KEYWORDS = {
    "def",
    "class",
    "import",
    "imports",
    "function",
    "return",
    "async",
    "await",
    "interface",
    "struct",
    "trait",
    "fn",
    "var",
    "let",
    "const",
}

# Crude multi-hop / named-entity-bridge signal: two or more capitalised tokens
# inside a single query suggest "X relates to Y" style synthesis, where DCI
# narrows the candidate space (grep) and vector then expands semantically.
_PROPER_NOUN = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")
_BRIDGE_WORDS = {
    "between",
    "and",
    "vs",
    "versus",
    "connect",
    "connects",
    "connecting",
    "relate",
    "relates",
    "relationship",
    "depend",
    "depends",
    "compare",
    "compared",
    "linked",
}

# Word counts that put a query in "long / paraphrastic" territory where vector
# hybrid is the safer bet (no exact-lexical signal to lock onto).
_LONG_QUERY_TOKENS = 14


class HeuristicRouter:
    """Pick a DCI / chained / vector strategy from cheap lexical signals.

    Scoring model: each rule adds a weight to one of three buckets — ``dci``,
    ``vector`` (the existing ``hybrid`` path), and ``bridge`` (chained mode).
    The highest-scoring bucket wins; ties resolve toward vector hybrid (the
    safer default). When ``bridge`` wins, the chain order is picked by which
    base bucket has more weight: ``dci`` heavier → ``dci_then_vector`` (grep
    first, then expand), ``vector`` heavier → ``vector_then_dci``.

    All weights are tunables registered in ``configs/components.yaml`` so the
    Phase 4 loop can evolve them per workload.
    """

    name = "heuristic_router"

    def __init__(
        self,
        *,
        quote_weight: float = 3.0,
        identifier_weight: float = 1.5,
        code_keyword_weight: float = 2.0,
        bridge_weight: float = 2.0,
        long_query_vector_weight: float = 1.5,
        dci_threshold: float = 1.0,
    ) -> None:
        self.quote_weight = quote_weight
        self.identifier_weight = identifier_weight
        self.code_keyword_weight = code_keyword_weight
        self.bridge_weight = bridge_weight
        self.long_query_vector_weight = long_query_vector_weight
        self.dci_threshold = dci_threshold

    @traced(span_name="retrieval.routers.heuristic")
    async def route(self, query: Query) -> RouteDecision:
        return self.decide(query.raw)

    # Sync entry point — exposed so unit tests / the Phase 4 evaluator can
    # score the heuristic table without an event loop. ``route`` wraps it.
    def decide(self, raw: str) -> RouteDecision:
        text = raw.strip()
        lower = text.lower()
        dci_score, vector_score, bridge_score = 0.0, 0.0, 0.0
        signals: list[str] = []

        if _QUOTED_PHRASE.search(text):
            dci_score += self.quote_weight
            signals.append("quoted")
        if _BACKTICKED.search(text):
            dci_score += self.code_keyword_weight
            signals.append("backticked")
        if _IDENTIFIER.search(text) or _PATH_LIKE.search(text) or _DOTTED_NAME.search(text):
            dci_score += self.identifier_weight
            signals.append("identifier")
        if any(tok in _CODE_KEYWORDS for tok in lower.split()):
            dci_score += self.code_keyword_weight
            signals.append("code_keyword")

        # Bridge: ≥ 2 proper nouns AND a relational connective word. This is
        # the "X relates to Y" pattern; grep nails one anchor, vector expands.
        proper_nouns = len(_PROPER_NOUN.findall(text))
        has_bridge_word = any(w in lower.split() for w in _BRIDGE_WORDS)
        if proper_nouns >= 2 and has_bridge_word:
            bridge_score += self.bridge_weight
            signals.append("entity_bridge")

        # Long queries → favour vector hybrid (paraphrastic territory). Added
        # even when a single DCI signal is present, because long prose with one
        # incidental identifier is still better served by semantic retrieval —
        # the chained-mode logic below sorts out ordering when both fire.
        if len(text.split()) >= _LONG_QUERY_TOKENS:
            vector_score += self.long_query_vector_weight
            signals.append("long_paraphrastic")

        strategy, intent, complexity = self._pick(dci_score, vector_score, bridge_score)
        reasoning = (
            f"signals={signals or ['none']} "
            f"scores={{dci:{dci_score:.1f},vector:{vector_score:.1f},bridge:{bridge_score:.1f}}}"
        )
        _log.info(
            "router.heuristic",
            strategy=strategy,
            intent=intent,
            dci=dci_score,
            vector=vector_score,
            bridge=bridge_score,
        )
        return RouteDecision(
            strategy=strategy,
            intent=intent,
            expected_complexity=complexity,
            reasoning=reasoning,
        )

    def _pick(
        self, dci: float, vector: float, bridge: float
    ) -> tuple[Strategy, Intent, Complexity]:
        # Bridge wins when both DCI and vector signals exist together AND the
        # bridge bucket itself is at least the threshold — otherwise pick the
        # heavier single-mode bucket. The "and" guards against false bridges
        # from incidental multi-noun queries with no DCI/vector signal.
        if bridge >= self.dci_threshold and dci > 0 and vector > 0:
            strategy: Strategy = "dci_then_vector" if dci >= vector else "vector_then_dci"
            return strategy, "relational", "high"
        # Plain bridge with no other signal — still favours chained DCI-first.
        if bridge >= self.dci_threshold:
            return "dci_then_vector", "relational", "med"
        if dci >= max(self.dci_threshold, vector):
            return "dci", "lookup", "low" if dci <= self.quote_weight else "med"
        # Vector hybrid is the safe default (paraphrastic + everything else).
        intent: Intent = "synthesis" if vector > 0 else "lookup"
        complexity: Complexity = "med" if vector > 0 else "low"
        return "hybrid", intent, complexity


__all__ = ["HeuristicRouter"]
