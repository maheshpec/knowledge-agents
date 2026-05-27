"""Evaluation dataset loading + frozen-set isolation (SPEC §9.1).

Three named splits live in ``configs/eval.yaml``:

* ``dev`` — seen freely by the self-improvement loop.
* ``rotating`` — shown only at generation boundaries.
* ``frozen`` — NEVER shown to the loop; final verification + manual audits only.

The loader enforces frozen isolation: while *evolution mode* is active (the
Phase 4 loop sets it via :func:`evolution_mode`), loading the frozen split raises
:class:`FrozenSetIsolationError`. Datasets are JSONL files of :class:`GoldQuery`,
resolved relative to the repo root.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from common.errors import ConfigError, FrozenSetIsolationError
from common.schemas import GoldQuery

# Repo root: …/loader.py -> datasets -> evaluation -> src -> <root>
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_CONFIG = REPO_ROOT / "configs" / "eval.yaml"

# Splits that must never reach the evolutionary loop (SPEC §9.1).
PROTECTED_SPLITS = frozenset({"frozen"})

# Process-wide flag toggled by the Phase 4 loop while it self-improves.
_evolution_mode: ContextVar[bool] = ContextVar("evolution_mode", default=False)


def is_evolution_mode() -> bool:
    """True while the self-improvement loop is actively evolving (SPEC §9.1)."""
    return _evolution_mode.get()


@contextmanager
def evolution_mode() -> Iterator[None]:
    """Mark the enclosed block as evolution mode; frozen loads raise inside it."""
    token = _evolution_mode.set(True)
    try:
        yield
    finally:
        _evolution_mode.reset(token)


class Dataset(BaseModel):
    """A named evaluation split: an ordered list of gold queries."""

    name: str
    queries: list[GoldQuery] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.queries)

    def subset(self, n: int) -> Dataset:
        """A deterministic prefix of ``n`` queries (used by the PR smoke run)."""
        return Dataset(name=f"{self.name}[:{n}]", queries=self.queries[:n])


def load_jsonl(path: str | Path) -> list[GoldQuery]:
    """Parse a JSONL file of :class:`GoldQuery` records."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise ConfigError(f"dataset file not found: {p}")
    queries: list[GoldQuery] = []
    for lineno, line in enumerate(p.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            queries.append(GoldQuery.model_validate(json.loads(line)))
        except Exception as exc:
            raise ConfigError(f"{p}:{lineno}: invalid GoldQuery record: {exc}") from exc
    return queries


def _dataset_path(name: str, config_path: str | Path) -> Path:
    cfg = OmegaConf.load(Path(config_path))
    datasets = cfg.get("datasets")  # type: ignore[union-attr]
    if datasets is None or name not in datasets:
        raise ConfigError(f"dataset '{name}' not declared in {config_path}")
    entry = datasets[name]
    path = entry.get("path") if hasattr(entry, "get") else entry
    if not path:
        raise ConfigError(f"dataset '{name}' has no 'path' in {config_path}")
    return Path(path)


def load_dataset(
    name: str,
    *,
    config_path: str | Path = DEFAULT_EVAL_CONFIG,
    allow_frozen: bool = False,
) -> Dataset:
    """Load a named split from ``configs/eval.yaml``, enforcing frozen isolation.

    Raises :class:`FrozenSetIsolationError` if a protected split (``frozen``) is
    requested while evolution mode is active, unless ``allow_frozen=True`` (the
    explicit, audited escape hatch for final pre-PR verification).
    """
    if name in PROTECTED_SPLITS and is_evolution_mode() and not allow_frozen:
        raise FrozenSetIsolationError(
            f"refusing to load protected split '{name}' during evolution mode "
            "(SPEC §9.1: the frozen set must never be shown to the loop)"
        )
    rel_path = _dataset_path(name, config_path)
    return Dataset(name=name, queries=load_jsonl(rel_path))


def corpus_dir(config_path: str | Path = DEFAULT_EVAL_CONFIG) -> Path:
    """Absolute path to the fixture corpus used for offline evaluation."""
    cfg = OmegaConf.load(Path(config_path))
    datasets = cfg.get("datasets")  # type: ignore[union-attr]
    rel = datasets.get("corpus") if datasets is not None else None
    if not rel:
        raise ConfigError(f"no 'datasets.corpus' declared in {config_path}")
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


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
