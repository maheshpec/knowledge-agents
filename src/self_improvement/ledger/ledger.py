"""Git-backed JSONL experiment ledger (SPEC §8.2.1).

Storage is append-only files under ``experiments/`` — one ``gen-NNN.jsonl`` per
generation, one JSON record per line, plus a per-run ``manifest.yaml`` and a
best-effort denormalized ``lineage.parquet`` for fast graph queries. Git is the
version control and audit trail; there is no database and no write coordination.

Records are append-only: an update writes a *new* full line for the experiment,
and readers fold by ``experiment_id`` with last-write-wins. This is crash-safe
(an interrupted run leaves ``status: running`` records visible) and diffable in
PRs, exactly as the SPEC argues for JSONL-over-database.

``replay`` re-executes an experiment end-to-end from its stored ``config`` via an
injected evaluator and (optionally) checks the result is within a noise band of
the original — the test that the record is complete and reproducible.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import yaml

from self_improvement.ledger.models import Experiment, RunManifest
from self_improvement.registry.pipeline_config import PipelineConfig

if TYPE_CHECKING:  # avoid importing the heavier runner module at import time
    from evaluation.runners.runner import EvalReport

# Re-runs an evaluation from a stored config and returns its report.
Evaluator = Callable[[PipelineConfig], Awaitable["EvalReport"]]


@runtime_checkable
class ExperimentLedger(Protocol):
    """The ledger contract consumed by the evolutionary loop (SPEC §8.2.1)."""

    async def append(self, exp: Experiment) -> None: ...
    async def update_status(self, experiment_id: str, status: str, **fields: Any) -> None: ...
    async def get(self, experiment_id: str) -> Experiment: ...
    async def lineage(self, experiment_id: str) -> list[Experiment]: ...
    async def query(self, predicate: Callable[[Experiment], bool]) -> list[Experiment]: ...
    async def replay(self, experiment_id: str) -> EvalReport: ...


class JSONLLedger:
    """File-backed :class:`ExperimentLedger` (SPEC §8.2.1 storage layout)."""

    def __init__(
        self,
        root: str | Path,
        *,
        evaluator: Evaluator | None = None,
        noise_band: float = 0.02,
    ) -> None:
        self.root = Path(root)
        self._evaluator = evaluator
        self.noise_band = noise_band
        self._lock = asyncio.Lock()
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

    # --- paths ----------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        d = self.root / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _gen_file(self, run_id: str, generation: int) -> Path:
        return self._run_dir(run_id) / f"gen-{generation:03d}.jsonl"

    def _iter_gen_files(self) -> Iterator[Path]:
        yield from sorted((self.root / "runs").glob("*/gen-*.jsonl"))

    # --- reads (fold append-only lines, last write wins) ----------------

    def _read_all(self) -> dict[str, Experiment]:
        folded: dict[str, Experiment] = {}
        for path in self._iter_gen_files():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                exp = Experiment.model_validate_json(line)
                folded[exp.experiment_id] = exp  # last occurrence wins
        return folded

    # --- writes ---------------------------------------------------------

    async def append(self, exp: Experiment) -> None:
        async with self._lock:
            path = self._gen_file(exp.run_id, exp.generation)
            with path.open("a") as fh:
                fh.write(exp.model_dump_json() + "\n")

    async def update_status(self, experiment_id: str, status: str, **fields: Any) -> None:
        async with self._lock:
            current = self._read_all().get(experiment_id)
            if current is None:
                raise KeyError(f"unknown experiment_id '{experiment_id}'")
            updated = current.model_copy(update={"status": status, **fields})
            path = self._gen_file(updated.run_id, updated.generation)
            with path.open("a") as fh:
                fh.write(updated.model_dump_json() + "\n")

    def write_manifest(self, manifest: RunManifest) -> None:
        """Write/overwrite ``runs/{run_id}/manifest.yaml`` (SPEC layout)."""
        path = self._run_dir(manifest.run_id) / "manifest.yaml"
        path.write_text(yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False))

    def read_manifest(self, run_id: str) -> RunManifest:
        path = self._run_dir(run_id) / "manifest.yaml"
        return RunManifest.model_validate(yaml.safe_load(path.read_text()))

    def flush_lineage(self) -> Path | None:
        """Best-effort denormalized lineage graph at ``experiments/lineage.parquet``.

        JSONL is the source of truth; the parquet is a fast-query cache the loop
        refreshes at generation boundaries. Silently skipped if pyarrow is absent.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:  # pragma: no cover - optional dep
            return None
        edges: list[dict[str, Any]] = []
        for exp in self._read_all().values():
            for parent in exp.parent_ids or [None]:
                edges.append(
                    {
                        "experiment_id": exp.experiment_id,
                        "parent_id": parent,
                        "run_id": exp.run_id,
                        "generation": exp.generation,
                        "status": exp.status,
                        "config_hash": exp.config_hash,
                    }
                )
        out = self.root / "lineage.parquet"
        pq.write_table(pa.Table.from_pylist(edges), out)
        return out

    # --- queries --------------------------------------------------------

    async def get(self, experiment_id: str) -> Experiment:
        exp = self._read_all().get(experiment_id)
        if exp is None:
            raise KeyError(f"unknown experiment_id '{experiment_id}'")
        return exp

    async def query(self, predicate: Callable[[Experiment], bool]) -> list[Experiment]:
        return [e for e in self._read_all().values() if predicate(e)]

    async def lineage(self, experiment_id: str) -> list[Experiment]:
        """Return the ancestry chain root→…→self (deduped, depth-first)."""
        all_exps = self._read_all()
        if experiment_id not in all_exps:
            raise KeyError(f"unknown experiment_id '{experiment_id}'")
        ordered: list[Experiment] = []
        seen: set[str] = set()

        def walk(eid: str) -> None:
            exp = all_exps.get(eid)
            if exp is None or eid in seen:
                return
            seen.add(eid)
            for parent in exp.parent_ids:
                walk(parent)
            ordered.append(exp)

        walk(experiment_id)
        return ordered

    # --- replay ---------------------------------------------------------

    async def replay(self, experiment_id: str) -> EvalReport:
        """Re-execute an experiment from its stored config (SPEC §8.2.1)."""
        if self._evaluator is None:
            raise RuntimeError("JSONLLedger.replay requires an evaluator")
        exp = await self.get(experiment_id)
        return await self._evaluator(exp.config)

    async def verify_replay(self, experiment_id: str, metric: str) -> bool:
        """True if a replay reproduces ``metric`` within ``noise_band`` (SPEC §8.2.1).

        Compares against the stored ``eval_results`` aggregated value for the
        metric. Raises if the original never recorded it.
        """
        exp = await self.get(experiment_id)
        original = _stored_metric(exp, metric)
        if original is None:
            raise ValueError(f"experiment '{experiment_id}' has no stored '{metric}' result")
        report = await self.replay(experiment_id)
        replayed = report.aggregated.get(metric)
        if replayed is None:
            return False
        return abs(replayed - original) <= self.noise_band


def _stored_metric(exp: Experiment, metric: str) -> float | None:
    """Pull a metric value from an experiment's stored eval_results, if present."""
    if not exp.eval_results:
        return None
    for result in exp.eval_results.values():
        if result.name == metric:
            return result.value
    # Fall back to the dict key (datasets keyed differently than metric name).
    direct = exp.eval_results.get(metric)
    return direct.value if direct else None


def load_jsonl(path: str | Path) -> list[Experiment]:
    """Parse a single ``gen-NNN.jsonl`` file into experiments (helper/CLI use)."""
    return [
        Experiment.model_validate_json(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


__all__ = ["ExperimentLedger", "JSONLLedger", "Evaluator", "load_jsonl"]
