"""Build a retrieval pipeline from registry components (SPEC §8.1).

:class:`PipelineConfig` is the serializable description of a full retrieval
pipeline — exactly the unit the evolutionary loop (§8.2) mutates. It mirrors the
``index.retrieval`` block of ``configs/default.yaml``.
:func:`pipeline_from_registry` realizes a config into a live
:class:`HybridPipeline` by pulling each component from the registry, so the loop
can never construct a component outside the declared search space.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from knowledge_index.retrieval import HybridPipeline, RRFFuser, WeightedFuser
from knowledge_index.retrieval.fusion.base import Fuser
from self_improvement.registry.registry import ComponentRegistry, RegistryDeps


class PipelineConfig(BaseModel):
    """A full retrieval-pipeline configuration (the evolutionary loop's genome)."""

    retrievers: list[str] = Field(default_factory=lambda: ["dense", "sparse_bm25"])
    fusion: str = "rrf"
    rrf_k: int = 60
    reranker: str = "cohere_rerank_3"
    reranker_top_k: int = 10
    post_processors: list[str] = Field(default_factory=lambda: ["mmr", "parent_expander"])
    mmr_lambda: float = 0.5
    query_ops: list[str] = Field(default_factory=lambda: ["rewrite"])


def _build_fuser(config: PipelineConfig) -> Fuser:
    if config.fusion in ("rrf", "hybrid_rrf"):
        return RRFFuser(k=config.rrf_k)
    if config.fusion == "weighted":
        return WeightedFuser()
    raise ValueError(f"unknown fusion strategy '{config.fusion}'")


def pipeline_from_registry(
    registry: ComponentRegistry,
    config: PipelineConfig,
    deps: RegistryDeps,
) -> HybridPipeline:
    """Realize ``config`` into a :class:`HybridPipeline` using registry components."""
    retrievers = [registry.instantiate("retrievers", name, deps=deps) for name in config.retrievers]
    reranker = registry.instantiate("rerankers", config.reranker, deps=deps)

    post = []
    for name in config.post_processors:
        params = {"lambda": config.mmr_lambda} if name == "mmr" else {}
        post.append(registry.instantiate("post_processors", name, params, deps=deps))

    query_ops = [registry.instantiate("query_ops", name, deps=deps) for name in config.query_ops]

    return HybridPipeline(
        retrievers=retrievers,
        reranker=reranker,
        fuser=_build_fuser(config),
        query_ops=query_ops,
        post_processors=post,
        rerank_k=config.reranker_top_k,
    )


__all__ = ["PipelineConfig", "pipeline_from_registry"]
