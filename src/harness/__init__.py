"""Harness entry point (SPEC §6.1, §14).

``answer()`` is the one-call public API: it wires Settings → the knowledge index
(Convoy B) → the retrieval pipeline (Convoy C) → the full Phase 2 orchestrator
graph and runs a single question to a cited :class:`GenerationResult`. By default
it uses the Phase 2 stack: a Plan-and-Execute planner, the query router, skills,
long-term memory, compaction, permission gates, and clean-context sub-agents.

All heavy wiring is imported lazily inside the functions so that importing light
submodules (``harness.cache``, ``harness.observability``) stays cheap and free of
import cycles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

if TYPE_CHECKING:
    from common.schemas import GenerationResult
    from harness.orchestrator.state import OrchestratorDeps

# Default per-request budget (USD) when the caller does not specify one.
DEFAULT_BUDGET_USD = 1.0

# Long-term memory lives in its own Qdrant collection (SPEC §6.3), not a payload
# namespace within the corpus collection.
MEMORY_COLLECTION = "ka_memory_longterm"


def build_default_deps(
    *,
    collection: str = "main",
    embedder_name: str = "voyage-3-large",
    planner_mode: Literal["react", "todo_list"] = "todo_list",
    enable_phase2: bool = True,
) -> OrchestratorDeps:
    """Construct production orchestrator deps from environment Settings.

    Uses the real Qdrant client, Voyage embeddings, the canonical hybrid pipeline
    (Convoy C), and the Anthropic-backed citation enforcer. When ``enable_phase2``
    (the default) it also wires the Phase 2 stack: a Plan-and-Execute planner, the
    LLM query router, the skill registry, long-term memory, compaction, permission
    gates, and a clean-context sub-agent handler.

    Requires API keys; for offline use, build :class:`OrchestratorDeps` yourself
    and pass via ``deps=``.
    """
    from common.settings import get_settings
    from harness.cache.embedding_cache import EmbeddingCache
    from harness.citation.enforcer import CitationEnforcer
    from harness.context.packer import DefaultPacker
    from harness.orchestrator.state import OrchestratorDeps
    from harness.planning.react import ReactPlanner
    from knowledge_index.embedding import build_embedder
    from knowledge_index.indexing import QdrantIndex
    from knowledge_index.retrieval import build_default_pipeline

    settings = get_settings()
    cache = EmbeddingCache(settings.embedding_cache_path)
    embedder = build_embedder(embedder_name, cache=cache)
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    index = QdrantIndex(collection, dim=embedder.dim, client=client)
    pipeline = build_default_pipeline(index, embedder)
    enforcer = CitationEnforcer(
        model="claude-sonnet-4-6", api_key=settings.anthropic_api_key or None
    )
    deps = OrchestratorDeps(
        pipeline=pipeline,
        enforcer=enforcer,
        packer=DefaultPacker(),
        planner=ReactPlanner(),
        planner_mode=planner_mode if enable_phase2 else "react",
    )
    if enable_phase2:
        _wire_phase2(deps, index=index, embedder=embedder, client=client)
    return deps


def _wire_phase2(deps: OrchestratorDeps, *, index: Any, embedder: Any, client: Any) -> None:
    """Attach the Phase 2 components to ``deps`` in place (SPEC §6.3-§6.10, §7.6.1)."""
    from pathlib import Path

    from common.config import load_config
    from harness.compaction.strategies import SelectiveRetentionCompactor
    from harness.memory.longterm import LongTermMemory
    from harness.memory.manager import LayeredMemory
    from harness.memory.session import SessionMemory
    from harness.memory.working import WorkingMemory
    from harness.permissions.gates import default_gates
    from harness.planning.todo import TodoListPlanner
    from harness.skills.registry import SkillRegistry
    from harness.subagents.orchestrator_agent import orchestrator_agent_fn
    from knowledge_index.indexing import QdrantIndex
    from knowledge_index.retrieval.routers.llm_router import LLMRouter

    cfg = load_config()

    # G — Plan-and-Execute planner + LLM query router.
    deps.todo_planner = TodoListPlanner()
    deps.router = LLMRouter()

    # F — skill registry: discover SKILL.md under the configured root (if any).
    skills_root = Path(str(cfg.get("skills", {}).get("root", "src/skills")))
    registry = SkillRegistry()
    if skills_root.exists():
        deps.skill_manifests = registry.discover(skills_root)
        deps.skills = registry
        deps.skill_k = int(cfg.get("skills", {}).get("select_k", 2))

    # E — layered memory; long-term lives in its own Qdrant collection.
    mem_index = QdrantIndex(MEMORY_COLLECTION, dim=embedder.dim, client=client)
    deps.memory = LayeredMemory(
        working=WorkingMemory(),
        session=SessionMemory("default"),
        long_term=LongTermMemory(mem_index, embedder),
    )

    # E — compaction (selective retention is the offline-safe default).
    deps.compactor = SelectiveRetentionCompactor()

    # F — permission gates from config thresholds.
    perms = cfg.get("permissions", {})
    budget_cfg = cfg.get("budget", {})
    from harness.permissions.base import Gate

    deps.gates = cast(
        "list[Gate]",
        default_gates(
            budget_threshold=float(budget_cfg.get("gate_budget_threshold", 0.25)),
            max_concurrent=int(perms.get("max_concurrent_subagents", 3)),
        ),
    )

    # E — clean-context sub-agent handler (built last; closes over the wired deps).
    deps.agent_fn = orchestrator_agent_fn(deps)


async def answer(question: str, **opts: Any) -> GenerationResult:
    """Answer ``question`` against the knowledge index, returning cited prose.

    Options (all optional):
        deps: a prebuilt OrchestratorDeps (skip Settings-based construction).
        app:  a prebuilt compiled orchestrator graph (skip building entirely).
        collection, embedder_name: passed to ``build_default_deps``.
        planner_mode: 'todo_list' (default, Phase 2) or 'react' (Phase 1 loop).
        budget_usd, k, max_hops, strictness, principals, thread_id.
    """
    from harness.orchestrator.graph import build_orchestrator, initial_state

    app = opts.get("app")
    if app is None:
        deps = opts.get("deps") or build_default_deps(
            collection=opts.get("collection", "main"),
            embedder_name=opts.get("embedder_name", "voyage-3-large"),
            planner_mode=opts.get("planner_mode", "todo_list"),
        )
        app = build_orchestrator(deps)

    state = initial_state(
        question,
        budget_usd=float(opts.get("budget_usd", DEFAULT_BUDGET_USD)),
        k=int(opts.get("k", 10)),
        max_hops=int(opts.get("max_hops", 1)),
        strictness=opts.get("strictness", "strict"),
        user_principals=opts.get("principals"),
    )
    config = {"configurable": {"thread_id": opts.get("thread_id", str(uuid4()))}}
    final = await app.ainvoke(state, config=config)
    return final["result"]


__all__ = ["answer", "build_default_deps", "DEFAULT_BUDGET_USD"]
