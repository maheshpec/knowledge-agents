"""Iterative / multi-hop retrieval (SPEC §7.6.7).

The agentic retrieval loop: retrieve, judge the evidence, follow up on gaps,
repeat until done / hops exhausted / budget spent. Exposed as the
``strategy='iterative'`` retriever (SPEC §7.6.1).
"""

from knowledge_index.retrieval.iterative.judge import (
    HopDecision,
    HopJudge,
    LLMHopJudge,
)
from knowledge_index.retrieval.iterative.retriever import (
    DEFAULT_HOP_COST_USD,
    DEFAULT_MAX_HOPS,
    IterativeRetriever,
)

__all__ = [
    "IterativeRetriever",
    "DEFAULT_MAX_HOPS",
    "DEFAULT_HOP_COST_USD",
    "HopDecision",
    "HopJudge",
    "LLMHopJudge",
]
