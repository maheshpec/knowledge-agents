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

from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from common.schemas import GenerationResult
    from harness.orchestrator.state import OrchestratorDeps

# Default per-request budget (USD) when the caller does not specify one.
DEFAULT_BUDGET_USD = 1.0


def build_default_deps(
    *,
    collection: str = "main",
    embedder_name: str = "voyage-3-large",
) -> OrchestratorDeps:
    """Construct production orchestrator deps from environment Settings.

    Uses the real Qdrant client, Voyage embeddings, the canonical hybrid pipeline
    (Convoy C), and the Anthropic-backed citation enforcer. Requires API keys; for
    offline use, build :class:`OrchestratorDeps` yourself and pass via ``deps=``.
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
    return OrchestratorDeps(
        pipeline=pipeline,
        enforcer=enforcer,
        packer=DefaultPacker(),
        planner=ReactPlanner(),
    )


async def answer(question: str, **opts: Any) -> GenerationResult:
    """Answer ``question`` against the knowledge index, returning cited prose.

    Options (all optional):
        deps: a prebuilt OrchestratorDeps (skip Settings-based construction).
        app:  a prebuilt compiled orchestrator graph (skip building entirely).
        collection, embedder_name: passed to ``build_default_deps``.
        budget_usd, k, max_hops, strictness, principals, thread_id.
    """
    from harness.orchestrator.graph import build_orchestrator, initial_state

    app = opts.get("app")
    if app is None:
        deps = opts.get("deps") or build_default_deps(
            collection=opts.get("collection", "main"),
            embedder_name=opts.get("embedder_name", "voyage-3-large"),
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
