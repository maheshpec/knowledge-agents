"""Orchestrator state + dependency container (SPEC §6.1).

``OrchestratorState`` is the LangGraph ``TypedDict`` checkpointed at every step.
It holds only serializable data; live components (pipeline, enforcer, packer,
budget tracker) live in :class:`OrchestratorDeps`, captured by node closures so
the graph stays picklable for ``SqliteSaver``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, runtime_checkable

from langchain_core.messages import BaseMessage

from common.schemas import (
    Citation,
    GenerationResult,
    Plan,
    Query,
    RetrievalCandidate,
    RetrievalResult,
)
from common.types import MemoryItem, Skill, SkillManifest
from harness.citation.base import CitedDraft, Strictness
from harness.citation.enforcer import CitationEnforcer
from harness.context.packer import DefaultPacker
from harness.planning.base import Planner

if TYPE_CHECKING:
    from harness.compaction.base import Compactor
    from harness.memory.manager import LayeredMemory
    from harness.permissions.base import Gate
    from harness.skills.registry import SkillRegistry
    from harness.subagents.base import AgentFn, SubAgentResult
    from knowledge_index.retrieval.routers.base import QueryRouter, RouteDecision
    from knowledge_index.retrieval.routers.pipeline import SupportsRetrieve as RouteRetrieve


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
    # --- Phase 2 (ka-2ba): routing, delegation, context-pack, permissions ---
    # The query router's verdict (intent/strategy/complexity), set once after plan.
    route_decision: RouteDecision | None
    # Skills selected for this query (SPEC §6.8) and long-term memory hits (§6.3),
    # rendered into the answer preamble by the packer.
    selected_skills: list[Skill]
    memory_hits: list[MemoryItem]
    # Sub-agent delegation (SPEC §6.4): results + a latch so we delegate once.
    subagent_results: list[SubAgentResult]
    delegated: bool
    delegation_depth: int
    # Permission gates (SPEC §6.10): the action a node intends + the gate outcome.
    pending_action: dict | None
    approval: dict | None
    approval_denied: bool
    active_subagents: int


@dataclass
class OrchestratorDeps:
    """Live components wired into the graph nodes (not checkpointed)."""

    pipeline: SupportsRetrieve
    enforcer: CitationEnforcer
    packer: DefaultPacker
    planner: Planner
    # Phase 2G: 'react' (default) plans one step at a time; 'todo_list' decomposes
    # the goal up front via ``todo_planner``. The plan node branches on this.
    planner_mode: Literal["react", "todo_list"] = "react"
    todo_planner: Planner | None = None
    system_prompt: str = "You are a helpful, citation-grounded research assistant."
    # token budget handed to the packer for the evidence block
    context_budget_tokens: int = 8000
    # nominal USD reserved per answer LLM call (settled to the real cost after)
    answer_cost_estimate: float = 0.05

    # --- Phase 2 components (ka-2ba). All optional: when None the graph behaves
    # exactly like the Phase 1 stack (no routing, delegation, skills, memory,
    # compaction, or gates), so Phase 1 callers and tests are unaffected. ---

    # Query router (G, SPEC §7.6.1): classifies intent/complexity and picks the
    # retrieval strategy. Separate from ``pipeline`` (which may itself be a
    # RouterPipeline); used here to decide whether the query warrants delegation.
    router: QueryRouter | None = None
    # Skills (F, SPEC §6.8): discovered manifests + the registry that selects the
    # top-k relevant skills for a query's intent.
    skills: SkillRegistry | None = None
    skill_manifests: list[SkillManifest] = field(default_factory=list)
    skill_k: int = 2
    # Memory (E, SPEC §6.3): long-term hits are read at the context-pack step.
    memory: LayeredMemory | None = None
    memory_k: int = 5
    # Compaction (E, SPEC §6.5): consulted on the ``compact?`` edge after observe.
    compactor: Compactor | None = None
    # Permission gates (F, SPEC §6.10): consulted before a sub-agent spawn.
    gates: list[Gate] | None = None
    # Sub-agents (E, SPEC §6.4): handler used to spawn clean-context delegates.
    agent_fn: AgentFn | None = None
    # Delegation policy: route a query to sub-agents when its intent is in this
    # set and a handler is wired. Disabled in spawned children to bound recursion.
    delegation_intents: tuple[str, ...] = ("comparison",)
    allow_delegation: bool = True
    # USD ceiling carved per delegated sub-agent.
    subagent_budget_usd: float = 0.25
    max_delegation_depth: int = 1

    extra: dict = field(default_factory=dict)

    def active_planner(self) -> Planner:
        """The planner the graph should use, per ``planner_mode`` (SPEC §6.2)."""
        if self.planner_mode == "todo_list" and self.todo_planner is not None:
            return self.todo_planner
        return self.planner


__all__ = ["OrchestratorState", "OrchestratorDeps", "SupportsRetrieve"]
