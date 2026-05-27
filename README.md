# Knowledge Agent Harness

A production-grade knowledge agent harness: an orchestration loop with planning,
memory, sub-agents, citations, and budgets, sitting over a swappable
knowledge-index (ingestion → chunking → enrichment → indexing → hybrid retrieval
→ reranking → packing), plus an AlphaEvolve-style self-improvement loop and an
evaluation framework.

See [`SPEC.md`](./SPEC.md) for the full design. This README covers getting started.

## Status

**Phase 2 — Agentic Loop** (current): the full orchestrator graph is wired —
query routing, a Plan-and-Execute planner, clean-context sub-agents, layered
memory, compaction, permission gates, and skills — over the Phase 1 hybrid
retrieval + citation foundation. See
[`docs/orchestrator-graph.md`](./docs/orchestrator-graph.md) for the node-by-node
graph and [`tests/integration/test_phase2_acceptance.py`](./tests/integration/test_phase2_acceptance.py)
for the end-to-end acceptance contract.

### Phase 2 capabilities

- **Query router (§7.6.1):** classifies each query's intent (lookup / synthesis /
  comparison / relational) and complexity, and picks a retrieval strategy. A
  `comparison` query is the trigger for sub-agent delegation.
- **Planner modes (§6.2):** `react` (Phase 1 — act one step at a time) or
  `todo_list` (Phase 2 default — decompose the goal into a dependency DAG and run
  independent steps concurrently). Select per request:
  `answer(q, planner_mode="react")`.
- **Sub-agents (§6.4):** the independent legs of a `todo_list` plan are delegated
  to clean-context sub-agents (each its own LangGraph instance, no parent
  history), spawned in parallel with carved child budgets; their citations are
  lifted back into the parent's evidence so provenance survives the boundary.
- **Memory (§6.3) + compaction (§6.5):** long-term hits are read at the
  context-pack step; history is compacted when it nears the model limit.
- **Skills (§6.8):** the registry selects the top-k skills for the query intent
  and the packer renders them into the answer preamble.
- **Permission gates (§6.10):** a `gate` node pauses the graph via LangGraph
  `interrupt` before a sensitive action (e.g. spawning beyond the concurrency
  cap). Thresholds come from `configs/default.yaml` (`permissions`, `budget`).

Every Phase 2 component is optional on `OrchestratorDeps`; with none wired the
graph collapses to the Phase 1 `plan → route → {retrieve | answer}` loop.

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

# query — defaults to the Phase 2 stack (todo_list planner + router + sub-agents)
uv run python -c "import asyncio; from harness import answer; \
  print(asyncio.run(answer('Compare X vs Y across cost and speed')).text)"

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
