# SPEC.md Compliance Audit — Whole-System

**Bead:** ka-eec · **Type:** audit (read/verify, no feature code)
**Audited branch:** `main` @ `c68f7225c71721ae85bb1099b58565fa1d62f925`
**Auditor:** polecat obsidian · **Date:** 2026-05-27

---

## Executive Summary

The repository **faithfully implements Phases 1 and 2** of SPEC.md (Foundation +
Agentic Loop) to a high standard — all 227 tests pass, ruff/mypy/format are clean,
and the code structure mirrors §4 closely. **Phases 3 and 4 are not implemented**:
the directories for iterative retrieval, GraphRAG, sandbox, and the entire
self-improvement system (evolutionary loop, ledger, reviewer, PR-gen, budget guard)
exist as empty `__init__.py` stubs, and several components are registered as
`NotImplementedError` placeholders so the registry/config can name them.

**VERDICT: SPEC is NOT fully implemented.** It is ~Phase-2-complete (roughly 50% of
the planned build). This is consistent with the repo's own phase log (most recent
commits are "Phase 2E–2H"). The spec's §10 phasing treats Phases 3–4 as in-scope for
"done," and the §11 whole-system acceptance criteria explicitly require self-improvement
(criterion 5) and advanced retrieval, which are absent.

---

## §10 Implementation Phase Status

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** (A–D) Foundation | schemas, ingestion, chunking (recursive/md), enrichment, embedding, Qdrant indexing, dense+BM25+RRF+rerank, MMR/parent post, rewriter, minimal orchestrator, budget/cache/observability/citation | ✅ **IMPLEMENTED** |
| **Phase 2** (E–I) Agentic Loop | plan-and-execute, sub-agents, memory (3 layers), compaction, permission gates, skills loader, query router + intent, eval framework | ✅ **IMPLEMENTED** |
| **Phase 3** Advanced Retrieval | iterative/multi-hop, full chunker zoo, all query-ops/post/rerankers, graph layer + GraphRetriever, sandbox | ❌ **MISSING** (empty stubs) |
| **Phase 4** Self-Improvement | registry (done), evolutionary loop, adversarial reviewer, PR-gen, budget guard, experiment ledger | ❌ **MISSING** (registry only) |

---

## Per-Section Compliance Table

| Spec § | Component | Status | Evidence |
|---|---|---|---|
| §4 | Repository structure | ✅ Matches | All dirs present; Phase 3/4 dirs are empty `__init__.py` placeholders |
| §5 | Core schemas | ✅ IMPLEMENTED | `src/common/schemas.py` — Source, Chunk, Query, RetrievalCandidate/Result, Citation, GenerationResult, Plan, PlanStep all present |
| §6.1 | Orchestrator (LangGraph) | ✅ IMPLEMENTED | `src/harness/orchestrator/graph.py`, `state.py`; plan→route→{answer/retrieve/subagent}→observe→compact?→finalize |
| §6.2 | Planning (ReAct + Plan-Execute) | ✅ IMPLEMENTED | `planning/react.py`, `planning/todo.py`, `base.py` |
| §6.3 | Memory (working/session/long-term) | ✅ IMPLEMENTED | `memory/working.py`, `session.py`, `longterm.py`, `extraction.py`, `manager.py` |
| §6.4 | Sub-agents (clean context) | ✅ IMPLEMENTED | `subagents/runner.py` — `asyncio.create_task`/`gather` parallelism (runner.py:80-81) |
| §6.5 | Compaction | ✅ IMPLEMENTED | `compaction/strategies.py`, wired at `graph.py:288` compact_node |
| §6.6 | Context packing | ✅ IMPLEMENTED | `context/packer.py` |
| §6.7 | Sandbox (Docker/firejail) | ❌ MISSING | `harness/sandbox/__init__.py` empty (Phase 3) |
| §6.8 | Skills loader | ✅ IMPLEMENTED | `skills/registry.py` |
| §6.9 | Observability | ✅ IMPLEMENTED | `observability/tracing.py` (`@traced`), `llm.py`, `logging.py` |
| §6.10 | Permissions / gates | ✅ IMPLEMENTED | `permissions/gates.py`, `graph.py` (interrupt-based) |
| §6.11 | Budget tracker | ✅ IMPLEMENTED | `budget/tracker.py`; orchestrator route is budget-aware |
| §6.12 | Cache (prompt/embed/retrieval) | ✅ IMPLEMENTED | `cache/prompt_cache.py`, `embedding_cache.py`, `retrieval_cache.py` |
| §6.13 | Citation enforcer | ✅ IMPLEMENTED | `citation/enforcer.py`, run at `finalize` node (graph.py) |
| §7.1 | Ingestion | ✅ IMPLEMENTED | `ingestion/parsers.py`, `dedup.py`, `normalize.py` |
| §7.2 | Chunking | ⚠️ PARTIAL | recursive, markdown_header, semantic present; **sentence_window, late_chunking, propositional MISSING** (Phase 3) |
| §7.3 | Enrichment | ✅ IMPLEMENTED | Null, Title, Contextual, Summary enrichers |
| §7.4 | Embedding | ✅ IMPLEMENTED | `embedding/embedders.py` |
| §7.5 | Indexing (Qdrant + ACL) | ✅ IMPLEMENTED | `indexing/qdrant_index.py`; ACL via payload filter (qdrant_index.py:154-171), public-or-intersect, no-leak default |
| §7.6.1 | Router | ✅ IMPLEMENTED | `routers/llm_router.py`, `pipeline.py`; `graph`/`iterative` strategies fall back to hybrid (Phase 3 stub) |
| §7.6.2 | Query ops | ⚠️ PARTIAL | Rewriter implemented; **HyDE, Decomposer, Stepback are `NotImplementedError` stubs** (Phase 3) |
| §7.6.3 | Retrievers | ⚠️ PARTIAL | Dense, SparseBM25 implemented; **GraphRetriever, ParentChildRetriever MISSING** (parent expansion exists as post-proc) |
| §7.6.4 | Fusion (RRF/weighted) | ✅ IMPLEMENTED | `fusion/rrf.py`, `weighted.py` |
| §7.6.5 | Reranking | ⚠️ PARTIAL | Null, Cohere implemented; **Voyage, LLM rerankers are `NotImplementedError` stubs** |
| §7.6.6 | Post-processing | ✅ IMPLEMENTED | MMR, ParentExpander, SpanExtractor, Deduplicator |
| §7.6.7 | Iterative/multi-hop | ❌ MISSING | `retrieval/iterative/__init__.py` empty (Phase 3) |
| §7.7 | Graph layer | ❌ MISSING | `knowledge_index/graph/__init__.py` empty (Phase 3) |
| §8.1 | Component registry | ✅ IMPLEMENTED | `self_improvement/registry/registry.py` (239 LOC), `spec.py`, `pipeline_config.py`; `configs/components.yaml` |
| §8.2 | Evolutionary loop | ❌ MISSING | `self_improvement/evolutionary/__init__.py` empty (Phase 4) |
| §8.2.1 | Experiment ledger | ❌ MISSING | `self_improvement/ledger/__init__.py` empty; `experiments/` has only `runs/.gitkeep`, **no `lineage.parquet`** |
| §8.3 | Adversarial reviewer | ❌ MISSING | `self_improvement/reviewer/__init__.py` empty (Phase 4) |
| §8.4 | PR generation | ❌ MISSING | `self_improvement/pr_gen/__init__.py` empty (Phase 4) |
| §8.5 | Budget guard | ❌ MISSING | `self_improvement/budget_guard/__init__.py` empty (Phase 4) |
| §9.1 | Datasets | ⚠️ PARTIAL | dev/rotating/frozen jsonl + 17-doc corpus present, but **51 queries each** vs spec's ~500/~500/~1000 (seed-scale) |
| §9.2 | Metrics | ✅ IMPLEMENTED | `metrics/retrieval.py` (recall/precision/nDCG/MRR/hit), `e2e.py`, `operational.py` |
| §9.3 | Runners | ✅ IMPLEMENTED | `runners/runner.py`, `orchestrator_runner.py` |
| §10.5 | Secrets/config hygiene | ✅ IMPLEMENTED | `common/settings.py` (pydantic-settings), `.env.example`, `.gitignore`, gitleaks in `.pre-commit-config.yaml` + CI |

---

## §11 Whole-System Acceptance Criteria

> These are end-to-end criteria. Most require live infrastructure (Qdrant, Voyage/Cohere,
> Anthropic, LangSmith) and the full ~1000-query frozen set; they **cannot be verified
> offline** in this sandbox. Status below reflects whether the capability exists in code
> and whether the criterion is *achievable* at the current phase.

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Cited answers, p50 <5s / p95 <15s | ⚠️ UNVERIFIED | Orchestrator + citation path exists; latency requires live run with real services |
| 2 | Citation precision >0.9, recall >0.85 on frozen set | ⚠️ UNVERIFIED | Citation/LLM-judge metrics coded; frozen set is 51 synthetic queries, needs live LLM |
| 3 | Retrieval recall@10 >0.85 on frozen set | ⚠️ UNVERIFIED | Recall@k metric coded; needs live index + full frozen set |
| 4 | Sub-agents parallelize (slowest, not sum) | ✅ SUPPORTED | `asyncio.gather` in `subagents/runner.py:80-81`; covered by `test_phase2_acceptance.py` |
| 5 | Self-improvement loop produced ≥1 merged PR improving frozen metric | ❌ **NOT MET** | Evolutionary loop / reviewer / PR-gen are unimplemented (Phase 4). Structurally impossible today |
| 6 | No PII/ACL leaks in red-team suite | ⚠️ PARTIAL | ACL filter enforced at index layer (no-leak default); **no dedicated red-team test suite found** |
| 7 | Every claim traceable to a chunk in LangSmith | ⚠️ SUPPORTED | `@traced` spans + citation enforcer; full LangSmith traceability needs live tracing |
| 8 | Total cost per query reported | ✅ SUPPORTED | Cost tracked in `budget/tracker.py` + orchestrator state/graph |

**Criterion 5 alone makes the whole-system "done" bar unreachable at the current phase.**

---

## Test / Lint / Typecheck Results

Run on audited HEAD (`c68f722`), local `.venv`:

| Gate | Command | Result |
|---|---|---|
| Tests | `pytest -q` | ✅ **227 passed**, 1 warning (ragas deprecation), 17.2s |
| Lint | `ruff check .` | ✅ All checks passed |
| Format | `ruff format --check src tests` | ✅ 164 files already formatted |
| Typecheck | `mypy` (configured) | ✅ Success: no issues in 51 source files |

**Caveat on mypy scope:** `pyproject.toml [tool.mypy] packages = ["common", "harness"]`,
so mypy type-checks only those 2 packages (51 files). `knowledge_index`,
`self_improvement`, and `evaluation` are **not** typechecked in CI. This is a real
coverage gap, though not a spec violation per se.

Integration tests with the `integration` marker (requiring live Qdrant/network) are
excluded by CI (`-m "not integration"`); they were not exercised here.

---

## Gaps / Missing Items (Actionable)

**Phase 3 (Advanced Retrieval) — entirely missing:**
1. Iterative / multi-hop retriever (§7.6.7) — empty dir.
2. GraphRAG: graph builder + GraphRetriever + Neo4j/networkx (§7.7) — empty dir.
3. Sandbox for tool isolation (§6.7) — empty dir. (§13 anti-pattern: "running tools without sandbox in production.")
4. Chunkers: `sentence_window`, `late_chunking`, `propositional` (§7.2).
5. Query ops: HyDE, Decomposer, Stepback — `NotImplementedError` stubs (§7.6.2).
6. Rerankers: Voyage, LLM — `NotImplementedError` stubs (§7.6.5).
7. Retrievers: GraphRetriever, ParentChildRetriever (§7.6.3).

**Phase 4 (Self-Improvement) — only the registry exists:**
8. Evolutionary loop with mutation/crossover (§8.2).
9. Experiment ledger (JSONL+git, replay) + `experiments/lineage.parquet` (§8.2.1).
10. Adversarial reviewer (§8.3).
11. PR generator (§8.4).
12. Budget guard / kill switch (§8.5).
13. `scripts/self_improve_run.py` is an explicit `NotImplementedError` stub.

**Cross-cutting:**
14. Eval datasets are seed-scale (51 queries each) vs spec's ~500/~500/~1000.
15. No `tests/eval/` directory (spec §4 lists `tests/{unit,integration,eval}`).
16. No red-team / ACL-leak test suite (needed for §11 criterion 6).
17. mypy CI coverage excludes `knowledge_index`, `self_improvement`, `evaluation`.

---

## Overall Verdict

**SPEC.md is NOT fully implemented.** The build is solid and well-tested through
**Phase 2 (Agentic Loop)** — schemas, full harness control loop, planning, memory,
sub-agents, compaction, permissions, skills, hybrid retrieval with rerank+citations,
the component registry, and the evaluation framework are all present and green.

**Phase 3 (Advanced Retrieval)** and **Phase 4 (Self-Improvement)** are unbuilt
(empty package stubs + `NotImplementedError` placeholders). Because §10 includes
those phases in the definition of "done" and §11 criterion 5 mandates a real
self-improvement merged PR, the whole-system acceptance bar is **not met**.

If the question is "is the *current phase* (2) correctly and fully implemented?" —
the evidence says **yes**. If it is "does main fully implement the entire SPEC?" —
the answer is **no**, by roughly two phases of work.
