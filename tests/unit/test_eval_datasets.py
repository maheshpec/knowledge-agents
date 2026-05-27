"""Tests for evaluation dataset loading + frozen-set isolation (SPEC §9.1)."""

from __future__ import annotations

import json

import pytest

from common.errors import ConfigError, FrozenSetIsolationError
from evaluation.datasets import (
    Dataset,
    evolution_mode,
    is_evolution_mode,
    load_dataset,
    load_jsonl,
)
from evaluation.datasets.loader import REPO_ROOT


def _write_eval_config(tmp_path, dev_path, frozen_path):
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(f"datasets:\n  dev:\n    path: {dev_path}\n  frozen:\n    path: {frozen_path}\n")
    return cfg


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_load_dataset_reads_jsonl(tmp_path):
    dev = tmp_path / "dev.jsonl"
    _write_jsonl(dev, [{"query_id": "q1", "query": "hello", "relevant_doc_ids": ["d1"]}])
    cfg = _write_eval_config(tmp_path, dev, tmp_path / "frozen.jsonl")

    ds = load_dataset("dev", config_path=cfg)
    assert isinstance(ds, Dataset)
    assert len(ds) == 1
    assert ds.queries[0].query == "hello"


def test_frozen_isolation_blocks_in_evolution_mode(tmp_path):
    frozen = tmp_path / "frozen.jsonl"
    _write_jsonl(frozen, [{"query_id": "f1", "query": "secret"}])
    cfg = _write_eval_config(tmp_path, tmp_path / "dev.jsonl", frozen)

    # outside evolution mode: allowed
    assert len(load_dataset("frozen", config_path=cfg)) == 1

    # inside evolution mode: refused
    with evolution_mode():
        assert is_evolution_mode() is True
        with pytest.raises(FrozenSetIsolationError):
            load_dataset("frozen", config_path=cfg)
        # explicit audited escape hatch still works
        assert len(load_dataset("frozen", config_path=cfg, allow_frozen=True)) == 1

    assert is_evolution_mode() is False


def test_missing_dataset_raises(tmp_path):
    cfg = _write_eval_config(tmp_path, tmp_path / "dev.jsonl", tmp_path / "frozen.jsonl")
    with pytest.raises(ConfigError):
        load_dataset("nonexistent", config_path=cfg)


def _seed(name):
    return load_jsonl(REPO_ROOT / "evaluation" / "datasets" / "seed" / f"{name}.jsonl")


def test_seed_datasets_are_committed_and_valid():
    # The generated seed splits must load and carry gold doc + chunk ids so
    # retrieval/citation metrics have something to score against (acceptance).
    for name in ("dev", "frozen", "rotating"):
        queries = _seed(name)
        assert len(queries) >= 50
        assert all(q.relevant_doc_ids for q in queries)
        assert all(q.relevant_chunk_ids for q in queries)
        # query ids are unique within a split
        assert len({q.query_id for q in queries}) == len(queries)


def test_seed_datasets_meet_spec_scale():
    # SPEC §9/§11 targets (Gap G1 / ka-94g): ~500 dev, ~500 rotating, ~1000 frozen.
    assert len(_seed("dev")) == 500
    assert len(_seed("rotating")) == 500
    assert len(_seed("frozen")) == 1000


def test_frozen_queries_are_disjoint_from_dev_and_rotating():
    # Frozen hold-out discipline (SPEC §13): the frozen *query strings* must never
    # coincide with anything the loop can see, so they cannot leak into search.
    seen = {q.query for q in _seed("dev")} | {q.query for q in _seed("rotating")}
    frozen = {q.query for q in _seed("frozen")}
    assert seen.isdisjoint(frozen)


def test_seed_splits_preserve_difficulty_stratification():
    # Every split must carry all three difficulty bands (SPEC §11 stratification).
    for name in ("dev", "rotating", "frozen"):
        bands = {q.difficulty for q in _seed(name)}
        assert bands == {"easy", "medium", "hard"}
