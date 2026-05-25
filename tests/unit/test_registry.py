"""Tests for the component registry: load, validate, sample, instantiate (SPEC §8.1)."""

import random

import pytest

from self_improvement.registry import (
    ComponentRegistry,
    PipelineConfig,
    RegistryDeps,
    RegistryError,
    pipeline_from_registry,
)

# --- offline fakes for components needing live deps ---


class _FakeIndex:
    async def search_dense(self, vec, k, filters):
        return []

    async def search_sparse(self, query, k, filters):
        return []


class _FakeEmbedder:
    async def embed_query(self, text):
        return [0.0, 1.0, 0.0]


class _FakeCohere:
    async def rerank(self, *, model, query, documents, top_n):
        class _R:
            results: list = []

        return _R()


async def _fetch_parent(_pid):
    return None


async def _complete(_prompt):
    return "rewritten"


def _deps() -> RegistryDeps:
    return RegistryDeps(
        index=_FakeIndex(),
        embedder=_FakeEmbedder(),
        cohere_client=_FakeCohere(),
        fetch_parent=_fetch_parent,
        completer=_complete,
    )


@pytest.fixture
def registry() -> ComponentRegistry:
    return ComponentRegistry.from_yaml()


# --- loading + introspection ---


def test_loads_all_categories(registry):
    assert set(registry.categories()) == {
        "chunkers",
        "enrichers",
        "retrievers",
        "rerankers",
        "post_processors",
        "query_ops",
    }


def test_list_and_get(registry):
    names = {s.name for s in registry.list("rerankers")}
    assert "cohere_rerank_3" in names
    spec = registry.get("chunkers", "recursive")
    assert spec.params["chunk_size"].default == 500


def test_unknown_category_and_component_raise(registry):
    with pytest.raises(RegistryError):
        registry.list("nope")
    with pytest.raises(RegistryError):
        registry.get("chunkers", "nope")


# --- param validation + sampling ---


def test_validate_params_fills_defaults(registry):
    spec = registry.get("chunkers", "recursive")
    validated = registry.validate_params(spec, {"chunk_size": 800})
    assert validated["chunk_size"] == 800
    assert validated["chunk_overlap"] == 75  # default filled


def test_validate_params_rejects_out_of_range(registry):
    spec = registry.get("post_processors", "mmr")
    with pytest.raises(RegistryError):
        registry.validate_params(spec, {"lambda": 2.0})  # range [0, 1]


def test_validate_params_rejects_unknown_key(registry):
    spec = registry.get("chunkers", "recursive")
    with pytest.raises(RegistryError):
        registry.validate_params(spec, {"bogus": 1})


def test_validate_enum_param(registry):
    spec = registry.get("rerankers", "llm_rerank")
    assert registry.validate_params(spec, {"model": "sonnet"})["model"] == "sonnet"
    with pytest.raises(RegistryError):
        registry.validate_params(spec, {"model": "gpt"})


def test_sample_params_within_range(registry):
    spec = registry.get("chunkers", "recursive")
    rng = random.Random(42)
    for _ in range(20):
        sampled = registry.sample_params(spec, rng)
        assert 200 <= sampled["chunk_size"] <= 1500
        assert 0 <= sampled["chunk_overlap"] <= 200


# --- instantiation ---


def test_instantiate_every_declared_component(registry):
    """Acceptance: instantiate every component declared in components.yaml."""
    deps = _deps()
    for category in registry.categories():
        for spec in registry.list(category):
            component = registry.instantiate(category, spec.name, deps=deps)
            assert component is not None


def test_instantiate_applies_params(registry):
    fuser = registry.instantiate("retrievers", "hybrid_rrf", {"rrf_k": 80})
    assert fuser._k == 80
    mmr = registry.instantiate("post_processors", "mmr", {"lambda": 0.2}, deps=_deps())
    assert mmr._lambda == 0.2


def test_instantiate_missing_dep_raises(registry):
    # dense needs index + embedder; with empty deps it must fail clearly.
    with pytest.raises(RegistryError):
        registry.instantiate("retrievers", "dense", deps=RegistryDeps())


# --- pipeline_from_registry ---


def test_pipeline_from_registry_builds_hybrid_pipeline(registry):
    from knowledge_index.retrieval import HybridPipeline

    config = PipelineConfig(
        retrievers=["dense", "sparse_bm25"],
        reranker="null",
        post_processors=["mmr"],
        query_ops=["rewrite"],
        rrf_k=70,
    )
    pipeline = pipeline_from_registry(registry, config, _deps())
    assert isinstance(pipeline, HybridPipeline)


@pytest.mark.asyncio
async def test_pipeline_from_registry_retrieves(registry):
    config = PipelineConfig(
        retrievers=["dense", "sparse_bm25"],
        reranker="null",
        post_processors=[],
        query_ops=[],
    )
    pipeline = pipeline_from_registry(registry, config, _deps())
    from common.schemas import Query

    result = await pipeline.retrieve(Query(raw="anything"), k=5)
    # Fake index returns no hits, but the pipeline composes and runs end to end.
    assert result.candidates == []
