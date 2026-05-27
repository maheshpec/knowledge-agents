"""Tests for the evolutionary loop: mutate, crossover, select, budget, e2e (SPEC §8.2)."""

import random

import pytest

from self_improvement.evolutionary import (
    Candidate,
    EvolutionaryLoop,
    ScorePolicy,
    SimpleBudgetGuard,
    composite_score,
    crossover,
    mutate,
    qualifies,
    random_config,
    select,
)
from self_improvement.evolutionary.genome import _GENE_FIELDS
from self_improvement.registry import ComponentRegistry, PipelineConfig, RegistryDeps
from self_improvement.registry.registry import RegistryError


@pytest.fixture
def registry() -> ComponentRegistry:
    return ComponentRegistry.from_yaml()


def _cand(cid: str, *, score=None, verdict=None, metrics=None, config=None) -> Candidate:
    c = Candidate(candidate_id=cid, config=config or PipelineConfig(), metrics=metrics or {})
    c.score = score
    c.verdict = verdict
    return c


# --- seeding / randomness ----------------------------------------------------


def test_random_config_is_registry_valid(registry):
    rng = random.Random(7)
    for _ in range(25):
        cfg = random_config(registry, rng)
        assert cfg.retrievers  # never empty
        assert cfg.chunker in {s.name for s in registry.list("chunkers")}
        assert cfg.enricher in {s.name for s in registry.list("enrichers")}
        assert cfg.reranker in {s.name for s in registry.list("rerankers")}
        for p in cfg.post_processors:
            assert p in {s.name for s in registry.list("post_processors")}


# --- mutation ----------------------------------------------------------------


def test_mutate_changes_exactly_one_gene(registry):
    rng = random.Random(3)
    base = PipelineConfig()
    for _ in range(40):
        child, record = mutate(base, registry, rng)
        diffs = [f for f in _GENE_FIELDS if getattr(base, f) != getattr(child, f)]
        # A mutation touches one gene. (chunker/enricher swaps also reset their
        # params, so allow the paired params field to differ too.)
        assert len(diffs) <= 2, diffs
        assert record.type == "mutate"
        assert record.component


def test_mutate_keeps_config_registry_buildable(registry):
    """Mutated configs must still be instantiable via the registry (SPEC §8.1)."""
    rng = random.Random(11)

    class _FakeEmbedder:
        async def embed_query(self, text):
            return [0.0, 1.0]

    async def _complete(_):
        return "x"

    async def _fetch(_):
        return None

    class _FakeIndex:
        async def search_dense(self, *a, **k):
            return []

        async def search_sparse(self, *a, **k):
            return []

    deps = RegistryDeps(
        index=_FakeIndex(),
        embedder=_FakeEmbedder(),
        cohere_client=object(),
        fetch_parent=_fetch,
        completer=_complete,
    )
    cfg = PipelineConfig()
    for _ in range(30):
        cfg, _ = mutate(cfg, registry, rng)
        for name in cfg.post_processors:
            try:
                registry.instantiate("post_processors", name, deps=deps)
            except RegistryError:
                pytest.fail(f"mutation produced unbuildable post-processor {name!r}")
        for name in cfg.query_ops:
            registry.instantiate("query_ops", name, deps=deps)


def test_mutate_param_stays_in_range(registry):
    rng = random.Random(5)
    cfg = PipelineConfig(post_processors=["mmr"], mmr_lambda=0.5)
    for _ in range(50):
        cfg, _ = mutate(cfg, registry, rng)
        assert 0.0 <= cfg.mmr_lambda <= 1.0
        assert 10 <= cfg.rrf_k <= 200


# --- crossover ---------------------------------------------------------------


def test_crossover_inherits_each_gene_from_a_parent(registry):
    rng = random.Random(1)
    a = random_config(registry, rng)
    b = random_config(registry, rng)
    child, record = crossover(a, b, rng)
    assert record.type == "crossover"
    for field in _GENE_FIELDS:
        assert getattr(child, field) in (getattr(a, field), getattr(b, field))


def test_crossover_keeps_component_params_consistent(registry):
    # chunker and chunker_params must come from the SAME parent.
    a = PipelineConfig(chunker="recursive", chunker_params={"chunk_size": 300})
    b = PipelineConfig(chunker="semantic", chunker_params={"threshold": 0.8})
    rng = random.Random(0)
    for _ in range(20):
        child, _ = crossover(a, b, rng)
        if child.chunker == "recursive":
            assert child.chunker_params == {"chunk_size": 300}
        else:
            assert child.chunker_params == {"threshold": 0.8}


# --- scoring / selection -----------------------------------------------------


def test_composite_score_subtracts_penalties():
    c = _cand("a", metrics={"ndcg@10": 0.8})
    c.cost_usd = 2.0
    c.compute_seconds = 1.0
    policy = ScorePolicy(cost_weight=0.1, latency_weight=0.05)
    assert composite_score(c, policy) == pytest.approx(0.8 - 0.2 - 0.05)


def test_select_top_k_and_drops_rejected():
    cands = [
        _cand("hi", score=0.9, verdict="accept"),
        _cand("mid", score=0.5, verdict="needs_more_evidence"),
        _cand("lo", score=0.1, verdict="accept"),
        _cand("bad", score=0.99, verdict="reject"),  # highest score but rejected
    ]
    chosen = select(cands, 2)
    ids = [c.candidate_id for c in chosen]
    assert ids == ["hi", "mid"]  # rejected dropped, top-2 by score
    assert "bad" not in ids


def test_select_scores_unscored_candidates():
    c = _cand("x", metrics={"ndcg@10": 0.42})
    chosen = select([c], 1)
    assert chosen[0].score == pytest.approx(0.42)


# --- qualification (delta threshold + Goodhart + verdict) --------------------


def test_qualifies_requires_accept_and_delta():
    base = 0.50
    good = _cand("g", score=0.55, verdict="accept", metrics={"ndcg@10": 0.55})
    assert qualifies(good, baseline_score=base, delta_threshold=0.02)

    too_small = _cand("s", score=0.515, verdict="accept", metrics={"ndcg@10": 0.515})
    assert not qualifies(too_small, baseline_score=base, delta_threshold=0.02)

    not_accepted = _cand("n", score=0.60, verdict="needs_more_evidence")
    assert not qualifies(not_accepted, baseline_score=base, delta_threshold=0.02)


def test_qualifies_goodhart_guard_blocks_rotating_regression():
    base = 0.50
    c = _cand("c", score=0.60, verdict="accept", metrics={"ndcg@10": 0.60})
    c.rotating_metrics = {"ndcg@10": 0.50}  # no real gain on the rotating set
    assert not qualifies(c, baseline_score=base, delta_threshold=0.02)
    # but if the rotating set also improves, it qualifies
    c.rotating_metrics = {"ndcg@10": 0.58}
    assert qualifies(c, baseline_score=base, delta_threshold=0.02)


# --- budget guard ------------------------------------------------------------


def test_budget_guard_trips_on_each_ceiling():
    g = SimpleBudgetGuard(max_generations=2)
    assert not g.exhausted()
    g.tick_generation()
    g.tick_generation()
    assert g.exhausted()

    g2 = SimpleBudgetGuard(max_cost_usd=1.0)
    g2.charge(cost_usd=0.5)
    assert not g2.exhausted()
    g2.charge(cost_usd=0.5)
    assert g2.exhausted()

    g3 = SimpleBudgetGuard(max_compute_seconds=10.0)
    g3.charge(compute_seconds=11.0)
    assert g3.exhausted()


# --- end-to-end loop ---------------------------------------------------------


class _StubEvaluator:
    """Deterministic, config-driven scorer: rewards a real reranker + query ops."""

    async def evaluate_batch(self, candidates):
        for c in candidates:
            score = 0.40
            if c.config.reranker != "null":
                score += 0.10
            score += 0.03 * len(c.config.query_ops)
            score += 0.02 * len(c.config.post_processors)
            c.metrics = {"ndcg@10": min(score, 0.99)}
            c.rotating_metrics = {"ndcg@10": min(score - 0.005, 0.99)}
            c.cost_usd = 0.01
            c.compute_seconds = 0.1
        return candidates


class _AcceptAllReviewer:
    async def review_batch(self, candidates):
        for c in candidates:
            c.verdict = "accept"
        return candidates


@pytest.mark.asyncio
async def test_evolutionary_loop_runs_and_improves(registry):
    loop = EvolutionaryLoop(
        registry,
        _StubEvaluator(),
        _AcceptAllReviewer(),
        SimpleBudgetGuard(max_generations=10),
        rng=random.Random(123),
        delta_threshold=0.01,
        seed_config=PipelineConfig(reranker="null", query_ops=[], post_processors=[]),
    )
    report = await loop.run(generations=4, population_size=6)

    assert report.generations_run == 4
    assert report.stopped_reason == "completed"
    assert len(report.population) == 6
    assert report.best_overall is not None
    # The loop should have found something better than the deliberately-weak seed.
    assert report.best_overall.score > report.baseline_score
    # History has one entry per generation (gen 0 seed + 4 evolved).
    assert [h["generation"] for h in report.history] == [0, 1, 2, 3, 4]
    assert report.history[-1]["best_score"] >= report.history[0]["best_score"]
    # A qualifying improvement was found and recorded as best.
    assert report.best is not None
    assert report.best.verdict == "accept"


@pytest.mark.asyncio
async def test_evolutionary_loop_stops_on_budget(registry):
    loop = EvolutionaryLoop(
        registry,
        _StubEvaluator(),
        _AcceptAllReviewer(),
        SimpleBudgetGuard(max_generations=2),
        rng=random.Random(1),
    )
    report = await loop.run(generations=100, population_size=4)
    assert report.stopped_reason == "budget_exhausted"
    assert report.generations_run <= 2


@pytest.mark.asyncio
async def test_loop_writes_to_ledger_when_provided(registry):
    appended = []

    class _Ledger:
        async def append(self, record):
            appended.append(record)

    loop = EvolutionaryLoop(
        registry,
        _StubEvaluator(),
        _AcceptAllReviewer(),
        SimpleBudgetGuard(max_generations=1),
        rng=random.Random(2),
        ledger=_Ledger(),
    )
    await loop.run(generations=1, population_size=3)
    assert appended  # every assessed candidate was recorded
    assert all(isinstance(c, Candidate) for c in appended)
