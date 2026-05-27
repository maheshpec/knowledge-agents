"""Genome operators: seed, mutate, crossover over :class:`PipelineConfig` (SPEC §8.2).

A genome is a full :class:`PipelineConfig`. Its mutable *genes* map onto config
fields and fall into three kinds:

- **choice**  — one component name from a registry category (chunker, enricher,
  reranker) or a small fixed set (fusion strategy).
- **subset**  — an ordered list drawn from a pool (retrievers, post-processors,
  query ops): mutation adds / drops / swaps one element.
- **param**   — a numeric parameter tuned in small steps within its declared
  range (rrf_k, mmr_lambda, reranker_top_k, and the selected chunker/enricher's
  own params).

All component names and parameter ranges come from the registry — the **only**
place the loop may pull components from (SPEC §8.1). Mutation changes exactly one
gene by a small step (SPEC §8.2 step 2); crossover recombines two parents field by
field at component boundaries (step 3).
"""

from __future__ import annotations

import random
from typing import Any

from self_improvement.evolutionary.types import MutationRecord
from self_improvement.registry.pipeline_config import PipelineConfig
from self_improvement.registry.registry import ComponentRegistry
from self_improvement.registry.spec import ParamSpec

# Genes whose choices/pools are not 1:1 with a registry category.
FUSION_CHOICES = ("rrf", "weighted")
RETRIEVER_POOL = ("dense", "sparse_bm25")  # hybrid_rrf is a fuser, not a retriever
RERANKER_TOP_K_RANGE = (1, 50)


# --- low-level steps ---------------------------------------------------------


def _other_choice(current: Any, choices: list[Any], rng: random.Random) -> Any:
    """Pick a choice different from ``current`` (or return current if it's the only one)."""
    alternatives = [c for c in choices if c != current]
    return rng.choice(alternatives) if alternatives else current


def _step_subset(current: list[str], pool: list[str], rng: random.Random) -> list[str]:
    """Add, drop, or swap one element — keeping order stable and avoiding dups."""
    items = list(current)
    addable = [p for p in pool if p not in items]
    ops: list[str] = []
    if addable:
        ops.append("add")
    if len(items) > 1:
        ops.append("drop")
    if items and addable:
        ops.append("swap")
    if not ops:
        return items
    op = rng.choice(ops)
    if op == "add":
        items.insert(rng.randint(0, len(items)), rng.choice(addable))
    elif op == "drop":
        items.pop(rng.randrange(len(items)))
    else:  # swap one element for one not present
        items[rng.randrange(len(items))] = rng.choice(addable)
    return items


def _random_subset(pool: list[str], rng: random.Random, *, allow_empty: bool) -> list[str]:
    lo = 0 if allow_empty else 1
    k = rng.randint(lo, len(pool))
    return rng.sample(pool, k)


def _step_param(spec: ParamSpec, value: Any, rng: random.Random) -> Any:
    """Nudge a numeric/enum param by one small step within its declared bounds."""
    if spec.type == "enum" and spec.values:
        return _other_choice(value, list(spec.values), rng)
    if spec.range is None:
        return value
    lo, hi = spec.range
    if spec.type == "int":
        cur = int(value)
        step = rng.choice([-1, 1])
        return max(int(lo), min(int(hi), cur + step))
    # float: step by ~10% of the range, clamped
    delta = (hi - lo) * 0.1
    return max(lo, min(hi, float(value) + rng.uniform(-delta, delta)))


# --- registry helpers --------------------------------------------------------


def _names(registry: ComponentRegistry, category: str) -> list[str]:
    return [s.name for s in registry.list(category)]


def _param_specs(registry: ComponentRegistry, category: str, name: str) -> dict[str, ParamSpec]:
    return dict(registry.get(category, name).params)


# --- seeding -----------------------------------------------------------------


def random_config(registry: ComponentRegistry, rng: random.Random) -> PipelineConfig:
    """Draw a random, valid genome from the registry search space (SPEC §8.2 seed)."""
    chunker = rng.choice(_names(registry, "chunkers"))
    enricher = rng.choice(_names(registry, "enrichers"))
    reranker = rng.choice(_names(registry, "rerankers"))
    post_pool = _names(registry, "post_processors")
    qop_pool = _names(registry, "query_ops")

    rrf_spec = _param_specs(registry, "retrievers", "hybrid_rrf").get("rrf_k")
    mmr_spec = _param_specs(registry, "post_processors", "mmr").get("lambda")

    return PipelineConfig(
        chunker=chunker,
        chunker_params=registry.sample_params(registry.get("chunkers", chunker), rng),
        enricher=enricher,
        enricher_params=registry.sample_params(registry.get("enrichers", enricher), rng),
        retrievers=_random_subset(list(RETRIEVER_POOL), rng, allow_empty=False),
        fusion=rng.choice(list(FUSION_CHOICES)),
        rrf_k=int(rrf_spec.sample(rng)) if rrf_spec else 60,
        reranker=reranker,
        reranker_top_k=rng.randint(*RERANKER_TOP_K_RANGE),
        post_processors=_random_subset(post_pool, rng, allow_empty=True),
        mmr_lambda=float(mmr_spec.sample(rng)) if mmr_spec else 0.5,
        query_ops=_random_subset(qop_pool, rng, allow_empty=True),
    )


# --- mutation ----------------------------------------------------------------


def _applicable_genes(config: PipelineConfig, registry: ComponentRegistry) -> list[str]:
    """Genes that can produce a real change given the current genome + registry."""
    genes: list[str] = ["fusion", "rrf_k", "reranker_top_k", "retrievers"]
    if len(_names(registry, "chunkers")) > 1:
        genes.append("chunker")
    if len(_names(registry, "enrichers")) > 1:
        genes.append("enricher")
    if len(_names(registry, "rerankers")) > 1:
        genes.append("reranker")
    if _param_specs(registry, "chunkers", config.chunker):
        genes.append("chunker_params")
    if _param_specs(registry, "enrichers", config.enricher):
        genes.append("enricher_params")
    if _names(registry, "post_processors"):
        genes.append("post_processors")
    if _names(registry, "query_ops"):
        genes.append("query_ops")
    if "mmr" in config.post_processors:
        genes.append("mmr_lambda")
    return genes


def _mutate_component_params(
    registry: ComponentRegistry,
    category: str,
    name: str,
    params: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    specs = _param_specs(registry, category, name)
    if not specs:
        return params
    merged = {**registry.get(category, name).defaults(), **params}
    pname = rng.choice(list(specs))
    merged[pname] = _step_param(specs[pname], merged.get(pname, specs[pname].default), rng)
    return merged


def mutate(
    config: PipelineConfig, registry: ComponentRegistry, rng: random.Random
) -> tuple[PipelineConfig, MutationRecord]:
    """Change exactly one gene by a small step (SPEC §8.2 step 2)."""
    gene = rng.choice(_applicable_genes(config, registry))
    updates: dict[str, Any] = {}

    if gene == "chunker":
        new = _other_choice(config.chunker, _names(registry, "chunkers"), rng)
        before = config.chunker
        updates = {"chunker": new, "chunker_params": {}}  # reset params to defaults
    elif gene == "enricher":
        new = _other_choice(config.enricher, _names(registry, "enrichers"), rng)
        before = config.enricher
        updates = {"enricher": new, "enricher_params": {}}
    elif gene == "reranker":
        new = _other_choice(config.reranker, _names(registry, "rerankers"), rng)
        before = config.reranker
        updates = {"reranker": new}
    elif gene == "fusion":
        new = _other_choice(config.fusion, list(FUSION_CHOICES), rng)
        before = config.fusion
        updates = {"fusion": new}
    elif gene == "retrievers":
        before = list(config.retrievers)
        new = _step_subset(config.retrievers, list(RETRIEVER_POOL), rng)
        updates = {"retrievers": new}
    elif gene == "post_processors":
        before = list(config.post_processors)
        new = _step_subset(config.post_processors, _names(registry, "post_processors"), rng)
        updates = {"post_processors": new}
    elif gene == "query_ops":
        before = list(config.query_ops)
        new = _step_subset(config.query_ops, _names(registry, "query_ops"), rng)
        updates = {"query_ops": new}
    elif gene == "rrf_k":
        spec = _param_specs(registry, "retrievers", "hybrid_rrf").get("rrf_k")
        before = config.rrf_k
        new = int(_step_param(spec, config.rrf_k, rng)) if spec else config.rrf_k
        updates = {"rrf_k": new}
    elif gene == "mmr_lambda":
        spec = _param_specs(registry, "post_processors", "mmr").get("lambda")
        before = config.mmr_lambda
        new = float(_step_param(spec, config.mmr_lambda, rng)) if spec else config.mmr_lambda
        updates = {"mmr_lambda": new}
    elif gene == "reranker_top_k":
        before = config.reranker_top_k
        lo, hi = RERANKER_TOP_K_RANGE
        new = max(lo, min(hi, config.reranker_top_k + rng.choice([-1, 1])))
        updates = {"reranker_top_k": new}
    elif gene == "chunker_params":
        before = dict(config.chunker_params)
        new = _mutate_component_params(
            registry, "chunkers", config.chunker, config.chunker_params, rng
        )
        updates = {"chunker_params": new}
    else:  # enricher_params
        before = dict(config.enricher_params)
        new = _mutate_component_params(
            registry, "enrichers", config.enricher, config.enricher_params, rng
        )
        updates = {"enricher_params": new}

    record = MutationRecord(type="mutate", component=gene, change={"before": before, "after": new})
    return config.model_copy(update=updates), record


# --- crossover ---------------------------------------------------------------

_GENE_FIELDS = (
    "chunker",
    "chunker_params",
    "enricher",
    "enricher_params",
    "retrievers",
    "fusion",
    "rrf_k",
    "reranker",
    "reranker_top_k",
    "post_processors",
    "mmr_lambda",
    "query_ops",
)


def crossover(
    parent_a: PipelineConfig, parent_b: PipelineConfig, rng: random.Random
) -> tuple[PipelineConfig, MutationRecord]:
    """Recombine two parents field-by-field at component boundaries (SPEC §8.2 step 3)."""
    updates: dict[str, Any] = {}
    inherited_from_b: list[str] = []
    for field in _GENE_FIELDS:
        if rng.random() < 0.5:
            updates[field] = getattr(parent_b, field)
            inherited_from_b.append(field)
        else:
            updates[field] = getattr(parent_a, field)
    # Keep chunker_params/enricher_params consistent with the chosen component:
    # if the component came from one parent, take that parent's params too.
    for comp, par in (("chunker", "chunker_params"), ("enricher", "enricher_params")):
        src = parent_b if comp in inherited_from_b else parent_a
        updates[comp] = getattr(src, comp)
        updates[par] = getattr(src, par)

    child = parent_a.model_copy(update=updates)
    record = MutationRecord(
        type="crossover", component="genome", change={"inherited_from_b": inherited_from_b}
    )
    return child, record


__all__ = [
    "FUSION_CHOICES",
    "RETRIEVER_POOL",
    "RERANKER_TOP_K_RANGE",
    "random_config",
    "mutate",
    "crossover",
]
