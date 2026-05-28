"""Component registry — the bounded search space (SPEC §8.1).

Loads ``configs/components.yaml`` into :class:`ComponentSpec` records and turns a
``(category, name, params)`` request into a concrete, configured component. It is
the **only** place the self-improvement loop may pull components from.

Components that need live wiring (an index, embedder, Cohere client, parent
fetcher, or LLM completer) receive it through :class:`RegistryDeps`, passed to
:meth:`instantiate`. Param-only components (chunkers, fusion-k, MMR-λ) need no
deps. A missing required dep raises :class:`RegistryError` with a clear message.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from common.config import CONFIGS_DIR
from self_improvement.registry.spec import ComponentSpec, ParamSpec, RegistryError

# Categories the registry understands (mirrors configs/components.yaml).
CATEGORIES = (
    "chunkers",
    "enrichers",
    "retrievers",
    "rerankers",
    "post_processors",
    "query_ops",
    # Phase 5 (SPEC §15.2): the DCI heuristic router exposes evolvable weights
    # so the Phase 4 loop can tune the DCI-vs-vector mix per workload.
    "routers",
)


@dataclass
class RegistryDeps:
    """Live wiring handed to component builders that need more than params.

    All optional: param-only components ignore this. A builder that needs a field
    left ``None`` raises :class:`RegistryError`.
    """

    index: Any = None  # Convoy B Index (search_dense/search_sparse)
    embedder: Any = None  # Embedder (embed_query)
    cohere_client: Any = None  # cohere.AsyncClientV2
    fetch_parent: Any = None  # async (parent_id) -> Chunk | None
    completer: Any = None  # async (prompt) -> str, for LLM query ops

    def require(self, field_name: str, *, for_component: str) -> Any:
        value = getattr(self, field_name)
        if value is None:
            raise RegistryError(
                f"component '{for_component}' requires deps.{field_name}, which was not provided"
            )
        return value


class ComponentRegistry:
    """Parse, validate, sample, and instantiate registered components (SPEC §8.1)."""

    def __init__(self, specs: dict[str, dict[str, ComponentSpec]]) -> None:
        # category -> {name -> spec}
        self._specs = specs

    # --- loading ---

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> ComponentRegistry:
        """Load the registry from ``configs/components.yaml`` (or an explicit path)."""
        cfg_path = Path(path) if path is not None else CONFIGS_DIR / "components.yaml"
        if not cfg_path.exists():
            raise RegistryError(f"components file not found: {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        specs: dict[str, dict[str, ComponentSpec]] = {}
        for category, entries in raw.items():
            specs[category] = {}
            for entry in entries or []:
                spec = ComponentSpec.from_yaml(category, entry)
                specs[category][spec.name] = spec
        return cls(specs)

    # --- introspection ---

    def categories(self) -> list[str]:
        return list(self._specs.keys())

    def list(self, category: str) -> list[ComponentSpec]:
        """Every component spec registered under ``category``."""
        if category not in self._specs:
            raise RegistryError(f"unknown category '{category}'; known: {sorted(self._specs)}")
        return list(self._specs[category].values())

    def get(self, category: str, name: str) -> ComponentSpec:
        try:
            return self._specs[category][name]
        except KeyError as exc:
            raise RegistryError(f"unknown component {category}/{name}") from exc

    # --- params ---

    def validate_params(self, spec: ComponentSpec, params: dict[str, Any]) -> dict[str, Any]:
        """Merge defaults, reject unknown keys, and bounds-check each value."""
        unknown = set(params) - set(spec.params)
        if unknown:
            raise RegistryError(
                f"unknown params for {spec.category}/{spec.name}: {sorted(unknown)}"
            )
        resolved = spec.defaults()
        for key, value in params.items():
            resolved[key] = value
        return {name: spec.params[name].validate(val) for name, val in resolved.items()}

    def sample_params(
        self, spec: ComponentSpec, rng: random.Random | None = None
    ) -> dict[str, Any]:
        """Draw a random param set within declared ranges (for Phase 4 sweeps)."""
        rng = rng or random.Random()
        return {name: pspec.sample(rng) for name, pspec in spec.params.items()}

    # --- instantiation ---

    def instantiate(
        self,
        category: str,
        name: str,
        params: dict[str, Any] | None = None,
        *,
        deps: RegistryDeps | None = None,
    ) -> Any:
        """Build a concrete, configured component for ``(category, name)``."""
        spec = self.get(category, name)
        validated = self.validate_params(spec, params or {})
        builder = _BUILDERS.get(category)
        if builder is None:
            raise RegistryError(f"no builder registered for category '{category}'")
        return builder(name, validated, deps or RegistryDeps())


# --- builders: (name, validated_params, deps) -> component -------------------


def _build_chunker(name: str, params: dict[str, Any], deps: RegistryDeps) -> Any:
    from knowledge_index.chunking import build_chunker

    return build_chunker(name, **params)


def _build_enricher(name: str, params: dict[str, Any], deps: RegistryDeps) -> Any:
    from knowledge_index.enrichment import build_enricher

    return build_enricher(name, **params)


def _build_retriever(name: str, params: dict[str, Any], deps: RegistryDeps) -> Any:
    from knowledge_index.retrieval import DenseRetriever, RRFFuser, SparseBM25Retriever

    if name == "dense":
        return DenseRetriever(
            deps.require("index", for_component=name),
            deps.require("embedder", for_component=name),
        )
    if name == "sparse_bm25":
        return SparseBM25Retriever(deps.require("index", for_component=name))
    if name == "hybrid_rrf":
        # The RRF fusion strategy that combines dense+sparse; rrf_k is the tunable.
        return RRFFuser(k=params.get("rrf_k", 60))
    if name == "iterative":
        # Multi-hop loop (SPEC §7.6.7) over a dense single-shot inner retriever,
        # with an LLM hop judge. max_hops is the tunable; the judge reuses the
        # shared completer dep so it runs offline under test.
        from knowledge_index.retrieval.iterative import IterativeRetriever, LLMHopJudge

        inner = DenseRetriever(
            deps.require("index", for_component=name),
            deps.require("embedder", for_component=name),
        )
        return IterativeRetriever(
            inner,
            LLMHopJudge(complete=deps.completer),
            max_hops=params.get("max_hops", 3),
        )
    raise RegistryError(f"no builder for retriever '{name}'")


def _build_reranker(name: str, params: dict[str, Any], deps: RegistryDeps) -> Any:
    from knowledge_index.retrieval import (
        CohereReranker,
        LLMReranker,
        NullReranker,
        VoyageReranker,
    )

    if name == "null":
        return NullReranker()
    if name == "cohere_rerank_3":
        return CohereReranker(client=deps.cohere_client)
    if name == "voyage_rerank_2":
        return VoyageReranker()
    if name == "llm_rerank":
        return LLMReranker()
    raise RegistryError(f"no builder for reranker '{name}'")


def _build_post(name: str, params: dict[str, Any], deps: RegistryDeps) -> Any:
    from knowledge_index.retrieval import (
        DeduplicatorPostProcessor,
        LostInTheMiddleReorder,
        MMRDiversifier,
        ParentExpander,
        SpanExtractor,
    )

    if name == "mmr":
        embedder = deps.require("embedder", for_component=name)
        return MMRDiversifier(embedder.embed_query, lambda_=params.get("lambda", 0.5))
    if name == "parent_expander":
        return ParentExpander(deps.require("fetch_parent", for_component=name))
    if name == "span_extractor":
        return SpanExtractor(max_sentences=params.get("max_sentences", 3))
    if name == "lost_in_the_middle":
        return LostInTheMiddleReorder()
    if name == "deduplicator":
        return DeduplicatorPostProcessor()
    raise RegistryError(f"no builder for post_processor '{name}'")


def _build_query_op(name: str, params: dict[str, Any], deps: RegistryDeps) -> Any:
    from knowledge_index.retrieval import Decomposer, HyDEExpander, Rewriter, Stepback

    if name == "rewrite":
        return Rewriter(complete=deps.completer)
    if name == "hyde":
        return HyDEExpander(complete=deps.completer)
    if name == "decompose":
        return Decomposer(complete=deps.completer)
    if name == "stepback":
        return Stepback(complete=deps.completer)
    raise RegistryError(f"no builder for query_op '{name}'")


def _build_router(name: str, params: dict[str, Any], deps: RegistryDeps) -> Any:
    from knowledge_index.retrieval.routers import HeuristicRouter

    if name == "heuristic":
        # Param-only (no LLM completer needed) — the heuristic router scores
        # cheap lexical signals and picks a DCI / chained / vector strategy.
        return HeuristicRouter(**params)
    raise RegistryError(f"no builder for router '{name}'")


_BUILDERS: dict[str, Callable[[str, dict[str, Any], RegistryDeps], Any]] = {
    "chunkers": _build_chunker,
    "enrichers": _build_enricher,
    "retrievers": _build_retriever,
    "rerankers": _build_reranker,
    "post_processors": _build_post,
    "query_ops": _build_query_op,
    "routers": _build_router,
}


__all__ = [
    "CATEGORIES",
    "RegistryDeps",
    "ComponentRegistry",
    "ComponentSpec",
    "ParamSpec",
    "RegistryError",
]
