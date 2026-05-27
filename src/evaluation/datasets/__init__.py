"""Evaluation datasets (SPEC §9.1): dev / rotating / frozen splits."""

from __future__ import annotations

from evaluation.datasets.loader import (
    PROTECTED_SPLITS,
    REPO_ROOT,
    Dataset,
    corpus_dir,
    evolution_mode,
    is_evolution_mode,
    load_dataset,
    load_jsonl,
)

__all__ = [
    "Dataset",
    "load_dataset",
    "load_jsonl",
    "corpus_dir",
    "evolution_mode",
    "is_evolution_mode",
    "REPO_ROOT",
    "PROTECTED_SPLITS",
]
