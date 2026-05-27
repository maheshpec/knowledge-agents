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


def test_seed_datasets_are_committed_and_valid():
    # The generated seed splits must load and carry gold doc ids (acceptance).
    for name in ("dev", "frozen", "rotating"):
        path = REPO_ROOT / "evaluation" / "datasets" / "seed" / f"{name}.jsonl"
        queries = load_jsonl(path)
        assert len(queries) >= 50
        assert all(q.relevant_doc_ids for q in queries)
