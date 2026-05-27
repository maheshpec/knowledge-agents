"""Tests for the experiment ledger (SPEC §8.2.1)."""

import pytest

from evaluation.metrics.base import MetricResult
from evaluation.runners.runner import EvalReport
from self_improvement.ledger import (
    Experiment,
    JSONLLedger,
    MutationRecord,
    RunManifest,
    config_hash,
    uuid7,
)
from self_improvement.registry.pipeline_config import PipelineConfig


def _exp(run_id="run-1", generation=0, **kw) -> Experiment:
    kw.setdefault("config", PipelineConfig())
    return Experiment(run_id=run_id, generation=generation, **kw)


# --- ids + hashing -----------------------------------------------------------


def test_uuid7_is_version7_unique_and_time_ordered():
    ids = [uuid7() for _ in range(200)]
    assert all(u.version == 7 for u in ids)
    assert len({str(u) for u in ids}) == 200  # unique
    timestamps = [int(str(u).replace("-", "")[:12], 16) for u in ids]
    assert timestamps == sorted(timestamps)  # non-decreasing ms timestamps


def test_config_hash_is_stable_and_content_addressed():
    a = PipelineConfig(reranker="cohere_rerank_3")
    b = PipelineConfig(reranker="cohere_rerank_3")
    c = PipelineConfig(reranker="voyage_rerank_2")
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)


def test_experiment_derives_config_hash_on_construction():
    exp = _exp()
    assert exp.config_hash == config_hash(PipelineConfig())


# --- append / get / query ----------------------------------------------------


async def test_append_get_roundtrip_and_storage_layout(tmp_path):
    ledger = JSONLLedger(tmp_path)
    exp = _exp(generation=1, mutation=MutationRecord(type="seed", component="chunker"))
    await ledger.append(exp)

    got = await ledger.get(exp.experiment_id)
    assert got.experiment_id == exp.experiment_id
    assert got.mutation and got.mutation.type == "seed"
    # SPEC storage layout: runs/{run_id}/gen-NNN.jsonl
    assert (tmp_path / "runs" / "run-1" / "gen-001.jsonl").exists()


async def test_get_unknown_raises(tmp_path):
    with pytest.raises(KeyError):
        await JSONLLedger(tmp_path).get("nope")


async def test_query_predicate(tmp_path):
    ledger = JSONLLedger(tmp_path)
    await ledger.append(_exp(generation=0, status="accepted"))
    await ledger.append(_exp(generation=1, status="rejected"))
    accepted = await ledger.query(lambda e: e.status == "accepted")
    assert len(accepted) == 1 and accepted[0].status == "accepted"


# --- append-only update_status (last write wins) -----------------------------


async def test_update_status_is_append_only_last_wins(tmp_path):
    ledger = JSONLLedger(tmp_path)
    exp = _exp(generation=2)
    await ledger.append(exp)
    await ledger.update_status(exp.experiment_id, "running")
    await ledger.update_status(exp.experiment_id, "evaluated", cost_usd=1.25)

    got = await ledger.get(exp.experiment_id)
    assert got.status == "evaluated"
    assert got.cost_usd == 1.25
    # Append-only: every transition is preserved on disk as its own line.
    lines = (tmp_path / "runs" / "run-1" / "gen-002.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3


async def test_update_unknown_raises(tmp_path):
    with pytest.raises(KeyError):
        await JSONLLedger(tmp_path).update_status("ghost", "running")


# --- manifest ----------------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    ledger = JSONLLedger(tmp_path)
    manifest = RunManifest(run_id="run-1", generations=5, population_size=8, dataset_refs=["dev"])
    ledger.write_manifest(manifest)
    assert (tmp_path / "runs" / "run-1" / "manifest.yaml").exists()
    loaded = ledger.read_manifest("run-1")
    assert loaded.generations == 5 and loaded.dataset_refs == ["dev"]


# --- lineage -----------------------------------------------------------------


async def test_lineage_walks_ancestry_in_order(tmp_path):
    ledger = JSONLLedger(tmp_path)
    root = _exp(generation=0)
    await ledger.append(root)
    child = _exp(generation=1, parent_ids=[root.experiment_id])
    await ledger.append(child)
    grandchild = _exp(generation=2, parent_ids=[child.experiment_id])
    await ledger.append(grandchild)

    chain = await ledger.lineage(grandchild.experiment_id)
    assert [e.experiment_id for e in chain] == [
        root.experiment_id,
        child.experiment_id,
        grandchild.experiment_id,
    ]


async def test_lineage_crossover_two_parents_deduped(tmp_path):
    ledger = JSONLLedger(tmp_path)
    p1, p2 = _exp(generation=0), _exp(generation=0)
    await ledger.append(p1)
    await ledger.append(p2)
    child = _exp(generation=1, parent_ids=[p1.experiment_id, p2.experiment_id])
    await ledger.append(child)
    chain = await ledger.lineage(child.experiment_id)
    ids = [e.experiment_id for e in chain]
    assert set(ids) == {p1.experiment_id, p2.experiment_id, child.experiment_id}
    assert ids[-1] == child.experiment_id  # self comes last


# --- replay round-trip (the completeness test, SPEC §8.2.1) ------------------


async def test_replay_reexecutes_from_stored_config(tmp_path):
    captured: dict[str, PipelineConfig] = {}

    async def evaluator(config: PipelineConfig) -> EvalReport:
        captured["config"] = config
        return EvalReport(dataset="dev", n=10, aggregated={"recall@5": 0.81})

    ledger = JSONLLedger(tmp_path, evaluator=evaluator)
    exp = _exp(
        config=PipelineConfig(reranker="voyage_rerank_2"),
        eval_results={"dev": MetricResult(name="recall@5", value=0.80)},
    )
    await ledger.append(exp)

    report = await ledger.replay(exp.experiment_id)
    # Replay re-ran the *stored* config, not a fresh default.
    assert captured["config"].reranker == "voyage_rerank_2"
    assert report.recall_at(5) == 0.81


async def test_verify_replay_within_noise_band(tmp_path):
    async def evaluator(config: PipelineConfig) -> EvalReport:
        return EvalReport(dataset="dev", n=10, aggregated={"recall@5": 0.815})

    ledger = JSONLLedger(tmp_path, evaluator=evaluator, noise_band=0.02)
    exp = _exp(eval_results={"dev": MetricResult(name="recall@5", value=0.80)})
    await ledger.append(exp)
    # 0.815 vs 0.80 → within 0.02 band.
    assert await ledger.verify_replay(exp.experiment_id, "recall@5") is True


async def test_verify_replay_outside_noise_band(tmp_path):
    async def evaluator(config: PipelineConfig) -> EvalReport:
        return EvalReport(dataset="dev", n=10, aggregated={"recall@5": 0.60})

    ledger = JSONLLedger(tmp_path, evaluator=evaluator, noise_band=0.02)
    exp = _exp(eval_results={"dev": MetricResult(name="recall@5", value=0.80)})
    await ledger.append(exp)
    assert await ledger.verify_replay(exp.experiment_id, "recall@5") is False


async def test_replay_without_evaluator_raises(tmp_path):
    ledger = JSONLLedger(tmp_path)
    exp = _exp()
    await ledger.append(exp)
    with pytest.raises(RuntimeError):
        await ledger.replay(exp.experiment_id)


# --- lineage.parquet best-effort export --------------------------------------


async def test_flush_lineage_writes_parquet(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    ledger = JSONLLedger(tmp_path)
    root = _exp(generation=0)
    await ledger.append(root)
    await ledger.append(_exp(generation=1, parent_ids=[root.experiment_id]))
    out = ledger.flush_lineage()
    assert out is not None and out.exists()
    table = pq.read_table(out)
    assert {"experiment_id", "parent_id", "generation"} <= set(table.column_names)
