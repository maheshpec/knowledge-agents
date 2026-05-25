"""Orchestrator state + dependency container (SPEC §6.1).

``OrchestratorState`` is the LangGraph ``TypedDict`` checkpointed at every step.
It holds only serializable data; live components (pipeline, enforcer, packer,
budget tracker) live in :class:`OrchestratorDeps`, captured by node closures so
the graph stays picklable for ``SqliteSaver``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypedDict, runtime_checkable

from langchain_core.messages import BaseMessage

from common.schemas import (
    Citation,
    GenerationResult,
    Plan,
    Query,
    RetrievalCandidate,
    RetrievalResult,
)
from harness.citation.base import CitedDraft, Strictness
from harness.citation.enforcer import CitationEnforcer
from harness.context.packer import DefaultPacker
from harness.planning.base import Planner


@runtime_checkable
class SupportsRetrieve(Protocol):
    """The slice of Convoy C's ``HybridPipeline`` the orchestrator consumes."""

    async def retrieve(self, query: Query, k: int) -> RetrievalResult: ...


class OrchestratorState(TypedDict, total=False):
    """Checkpointed orchestrator state (SPEC §6.1)."""

    question: str
    # Conversation history (Phase 2: multi-turn + compaction operate on this).
    messages: list[BaseMessage]
    plan: Plan | None
    retrieval_results: list[RetrievalResult]
    candidates: list[RetrievalCandidate]
    draft: CitedDraft | None
    result: GenerationResult | None
    citations: list[Citation]
    scratchpad: str
    # budget snapshot (the live tracker lives in deps, not in checkpointed state)
    budget_limit: float
    budget_remaining: float
    budget_exhausted: bool
    # control
    hops: int
    max_hops: int
    k: int
    strictness: Strictness
    user_principals: list[str]
    trace_id: str


@dataclass
class OrchestratorDeps:
    """Live components wired into the graph nodes (not checkpointed)."""

    pipeline: SupportsRetrieve
    enforcer: CitationEnforcer
    packer: DefaultPacker
    planner: Planner
    system_prompt: str = "You are a helpful, citation-grounded research assistant."
    # token budget handed to the packer for the evidence block
    context_budget_tokens: int = 8000
    # nominal USD reserved per answer LLM call (settled to the real cost after)
    answer_cost_estimate: float = 0.05
    extra: dict = field(default_factory=dict)


__all__ = ["OrchestratorState", "OrchestratorDeps", "SupportsRetrieve"]
