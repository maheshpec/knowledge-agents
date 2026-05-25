# Knowledge Agent Harness

A production-grade knowledge agent harness: an orchestration loop with planning,
memory, sub-agents, citations, and budgets, sitting over a swappable
knowledge-index (ingestion → chunking → enrichment → indexing → hybrid retrieval
→ reranking → packing), plus an AlphaEvolve-style self-improvement loop and an
evaluation framework.

See [`SPEC.md`](./SPEC.md) for the full design. This README covers getting started.

## Status

**Phase 1A — Foundation** (this checkpoint): repo skeleton, core Pydantic
schemas, typed settings, Hydra config plumbing, observability (`@traced` +
structured logging + LLM telemetry), budget tracker, and the three-tier cache.
Later phases build ingestion, retrieval, the orchestrator graph, and the
self-improvement loop on top of these primitives.

## Requirements

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency management

## Quick-start

```bash
uv sync                 # install dependencies into .venv
cp .env.example .env     # then fill in your API keys

# run the unit tests
uv run pytest tests/unit

# sanity-check the core schemas
uv run python -c "from common.schemas import Chunk; print(Chunk(chunk_id='x', doc_id='y', text='z'))"
```

Later phases add:

```bash
# ingest (Phase 1B)
uv run scripts/ingest.py --src ./docs --collection main

# query (Phase 1D)
uv run python -c "from harness import answer; print(answer('what is X?'))"

# eval (Phase 2)
uv run scripts/eval_run.py --dataset frozen --pipeline configs/default.yaml

# self-improvement (Phase 4)
uv run scripts/self_improve_run.py --generations 5 --population 8 --budget-usd 50
```

## Layout

```
src/
  common/           # schemas, types, errors, settings, config plumbing
  harness/          # orchestrator, planning, memory, observability, budget, cache, …
  knowledge_index/  # ingestion → retrieval pipeline
  self_improvement/ # registry, evolutionary loop, reviewer, PR gen
  evaluation/       # datasets, metrics, runners
configs/            # composable Hydra configs (default, components, eval, self_improvement)
tests/              # unit / integration / eval
experiments/        # versioned, git-backed experiment ledger
```

## Configuration

All secrets are read from the environment via `common.settings.Settings`
(`pydantic-settings`). No API keys live in the repo. `gitleaks` runs as a
pre-commit hook and in CI. See `SPEC.md §10.5`.

Pipeline and harness behavior is configured via composable Hydra YAML in
`configs/`. Load programmatically with `common.config.load_config("default")` or
override on the CLI (`uv run … budget.max_usd=5`).
