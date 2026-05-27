# Orchestrator Graph (Phase 2)

The orchestrator is a compiled LangGraph `StateGraph` (`src/harness/orchestrator/`).
Phase 2 (epic ka-2ba) extends the Phase 1 `plan → route → {retrieve | answer}`
loop with query routing, sub-agent delegation, a context-pack step, compaction,
and permission gates. Live components are held in `OrchestratorDeps` (captured by
node closures so the checkpointed `OrchestratorState` stays serializable); every
Phase 2 component is **optional** — when none are wired the graph reduces to the
Phase 1 loop, so Phase 1 callers and tests are unaffected.

## Graph

```
                 ┌──────┐
   START ───────▶│ plan │  active planner: react | todo_list (SPEC §6.2)
                 └──┬───┘
                    ▼
              ┌───────────┐
              │  context  │  router intent/complexity (§7.6.1)
              │           │  + skill select (§6.8) + memory read (§6.3)
              └────┬──────┘
                   ▼
              ┌─────────┐◀──────────────────────────────┐
              │  route  │  budget-aware decision          │
              └────┬────┘                                 │
        ┌──────────┼───────────────┐                      │
        ▼          ▼               ▼                      │
   ┌────────┐ ┌──────────┐   ┌──────────┐                 │
   │ answer │ │ retrieve │   │  gate?   │  permission     │
   │        │ │          │   │ (§6.10)  │  pause/deny     │
   └───┬────┘ └────┬─────┘   └────┬─────┘                 │
       │           │              ▼ (approved)            │
       │           │         ┌───────────┐                │
       │           │         │ sub-agent │  spawn_all,     │
       │           │         │  (§6.4)   │  lift citations │
       │           │         └────┬──────┘                │
       │           └──────┬───────┘                       │
       │                  ▼                               │
       │            ┌──────────┐                          │
       │            │ observe  │  digest + plan.adapt      │
       │            └────┬─────┘                           │
       │                 ▼                                 │
       │            ┌──────────┐   compact (§6.5)          │
       │            │ compact? │──────────────────────────┘
       │            └────┬─────┘  else ─────────────────▶ route
       ▼
  ┌──────────┐
  │ finalize │  citation enforcement (§6.13)
  └────┬─────┘
       ▼
      END
```

## Nodes

| Node | Purpose | Component (deps) |
|------|---------|------------------|
| `plan` | Build/refresh the plan. `todo_list` decomposes the goal into a dependency DAG; `react` emits a single act step. | `planner` / `todo_planner`, `planner_mode` |
| `context` | Run the query router once, then select skills for the query intent and read long-term memory hits for the answer preamble. Each branch is a no-op when its component is `None`. | `router`, `skills` + `skill_manifests`, `memory` |
| `route` | Budget-aware fan-out. Priority: **budget exhausted → answer**; **delegation warranted → sub-agent**; **hops remain → retrieve**; **else → answer**. | — |
| `retrieve` | One retrieval hop via the pipeline (which may itself be a `RouterPipeline`); accumulates candidates, debits budget, bumps `hops`. | `pipeline` |
| `gate` | (Only when `gates` configured.) Surfaces the intended spawn as a `pending_action` and evaluates the gates; trips → pause via `interrupt`; denial → skip the spawn. | `gates` |
| `sub-agent` | Delegate the plan's independent legs to clean-context sub-agents in parallel (`spawn_all`), carving child budgets; lift their citations back into the parent candidate set. | `agent_fn` |
| `observe` | Digest the hop (scratchpad note) and `plan.adapt` on the latest observation. | `planner` |
| `compact?` | (Only when `compactor` configured.) If history exceeds the threshold, route through `compact`; else back to `route`. | `compactor` |
| `answer` | Order evidence for the packer, prepend the skills + memory preamble, and draft a cited answer. Refuses if out of budget. | `packer`, `enforcer` |
| `finalize` | Enforce citations into the `GenerationResult` at the configured strictness. | `enforcer` |

## Delegation policy

A query is delegated to sub-agents when **all** hold (`_should_delegate`):

1. `agent_fn` is wired and `allow_delegation` is true;
2. it has not already delegated and `delegation_depth < max_delegation_depth`;
3. the router's `intent` is in `delegation_intents` (default `("comparison",)`);
4. the plan has ≥2 **independent** steps (no `depends_on`) *and* ≥1 dependent
   step — the independent legs become sub-agents, the dependent step (e.g. the
   synthesis) is performed by the parent's `answer` node.

Spawned children run with `allow_delegation=False` and `delegation_depth=1`, so
delegation never recurses unbounded (SPEC §6.4).

## Citation preservation across the boundary

Sub-agents return a `GenerationResult` whose citations point at the chunks they
grounded on. `subagent_node` rebuilds a candidate per citation — keeping the
original `chunk_id` (so the source pointer survives) and carrying the citation's
quote or the sub-agent's answer text as the body — and accumulates them into the
parent's candidate set. The parent can then re-cite that evidence, so provenance
is preserved end-to-end (the Phase 2 acceptance contract, SPEC §10).

## Entry point

`harness.answer(question, **opts)` builds the default Phase 2 stack
(`planner_mode="todo_list"`) via `build_default_deps`. Pass `planner_mode="react"`
for the Phase 1 loop, or `deps=`/`app=` to inject a prebuilt stack (as the tests
do, fully offline).
