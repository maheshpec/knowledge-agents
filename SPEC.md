# Knowledge Agent Harness — Build Spec

> A working spec to hand to Claude Code. Built to be incrementally implementable. Each module has an interface, a phase number, and acceptance criteria. Read the **Goals**, **Architecture**, and **Phases** sections first, then jump to the module you're building.

---

## 1. Goals

Build a production-grade knowledge agent harness composed of:

1. **Harness** — orchestration loop, planning, memory, sub-agents, compaction, sandboxing, skills, observability, permissions, budgets, caching, citations.
2. **Knowledge index** — ingestion → chunking → enrichment → indexing → retrieval → reranking → packing, with adaptive routing and iterative multi-hop loops.
3. **Self-improvement system** — an AlphaEvolve-style evolutionary loop that proposes, tests, and PRs improvements to the index, constrained to a known component registry.
4. **Evaluation framework** — retrieval metrics + end-to-end metrics + held-out test sets, so the self-improvement loop has a ground-truth target.

### Non-goals (initial release)

- Open-ended "agent reads arXiv and invents new retrieval techniques." Self-improvement is bounded to the component registry; novelty comes from humans extending the registry.
- Real-time updates on streaming data sources. Batched re-indexing only.
- On-device / edge deployment.
- Multi-tenant SaaS hosting. Single-tenant deployment with ACL enforcement.

### Design principles

- **LangGraph for control flow**, LangChain for primitives, LangSmith for tracing/eval. Plain Python everywhere else.
- **Every component is swappable.** No hard imports of concrete classes outside the registry layer.
- **Every component is benchmarkable.** Each retrieval-path module exposes a deterministic config dict so the self-improvement loop can sweep it.
- **Citations are first-class.** Every generated claim carries chunk-level provenance; the orchestrator enforces it.
- **Sub-agents have clean contexts.** Parent passes a task + return schema, never raw history.

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Chat Experience                                │
│  (Streaming UI, citation rendering, approvals UI — out of scope here)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                              Harness                                    │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │Orchestrator│──│ Planning │──│Sub-agents│──│ Context Engineering  │ │
│  │ (LangGraph)│  │          │  │          │  │ (packing/compaction) │ │
│  └─────┬──────┘  └──────────┘  └──────────┘  └──────────────────────┘ │
│        │                                                                │
│  ┌─────▼────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐│
│  │ Memory       │  │ Skills  │  │Permission│  │ Budget  │  │  Cache  ││
│  │ (short/long) │  │         │  │  /Gates  │  │ Tracker │  │         ││
│  └──────────────┘  └─────────┘  └──────────┘  └─────────┘  └─────────┘│
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Observability (LangSmith traces, OpenTelemetry spans)          │   │
│  └────────────────────────────────────────────────────────────────┘   │
└──────────┬─────────────────────────────────────────────────────────────┘
           │
┌──────────▼─────────────────────────────────────────────────────────────┐
│                              Tools                                      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                   Knowledge Index (primary tool)                  │ │
│  │  Router → {Vector | BM25 | Graph | Hybrid} → Rerank → Pack       │ │
│  │  with HyDE, query rewrite, decomposition, iterative loops        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────┐  ┌────────────────┐  ┌────────────────────┐│
│  │ Code Sandbox         │  │ Web Search     │  │ MCP Tools (varies) ││
│  │ (Docker/firejail)    │  │                │  │                    ││
│  └──────────────────────┘  └────────────────┘  └────────────────────┘│
└────────────────────────────────────────────────────────────────────────┘

       ┌──────────────────────────────────────────────────────────┐
       │             Self-Improvement Loop (offline)               │
       │   Registry → Mutate → Eval → Adversarial Review → PR     │
       └──────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | LangChain ecosystem |
| Orchestration | **LangGraph** ≥ 0.2 | Stateful, checkpointed, supports HITL, sub-graphs for sub-agents. Pin in `pyproject.toml`; checkpointer moved to `langgraph-checkpoint-sqlite` package — import via `from langgraph.checkpoint.sqlite import SqliteSaver` after installing both packages |
| Primitives | **LangChain** | Splitters, retrievers, embedders, ensemble retriever (RRF), document loaders |
| Observability | **LangSmith** + OpenTelemetry | Traces, eval runs, A/B comparisons |
| LLM | **Anthropic Claude** (primary) | `claude-sonnet-4-6` for generation/synthesis, `claude-haiku-4-5-20251001` for routing/classification/cheap calls. OpenAI fallback only if Anthropic unavailable. Prompt caching crucial for contextual retrieval (see §6.12) |
| Embeddings | **Voyage** (`voyage-3-large`, 1024-dim) primary; `text-embedding-3-large` (3072-dim) fallback | Configurable; bake-off on dev set before locking in |
| Vector store | **Qdrant** ≥ 1.10 | Hybrid (dense + sparse) in one collection via named vectors; payload filtering; snapshots |
| Sparse | **Qdrant + `fastembed` sparse model** (e.g. `Qdrant/bm25`) — note: Qdrant has no native BM25 statistics; `fastembed` ships a precomputed BM25-style sparse encoder. If recall is unacceptable on the dev set, fall back to `rank_bm25` over a parallel sidecar index. Phase 1 risk: bake off both during ingestion of dev corpus | |
| Reranker | **Cohere Rerank 3** or **Voyage rerank-2** | Cross-encoder, swappable via registry |
| Graph store | **Neo4j** (or `networkx` for dev) | For GraphRAG route |
| Sandbox | **Docker** containers via `docker-py`, with `firejail` fallback | Tool execution isolation |
| Validation | **Pydantic v2** | All schemas |
| Config | **Hydra** | Composable configs for sweeps |
| Testing | **pytest** + **ragas** + custom retrieval evals | |
| Package mgmt | **uv** | Fast, lockfile-based |

### 3.1 Considered alternatives (and why not)

- **MEOW / Gas City (Steve Yegge).** A Beads-based multi-agent orchestration framework with Dolt-backed git-versioned state. Genuinely interesting and the work-as-primitive model is a good one. Rejected for this project because (a) it targets teams running 10+ parallel agents and is explicitly counterproductive at smaller scales; (b) it's early-stage with significant cost and complexity; (c) it would create a second orchestration model running alongside LangGraph, which the harness itself uses — conceptual collision between "the system orchestrating the build" and "the system being built." We borrow one specific idea (git-versioned, replayable experiment records) in §8.2 without taking the framework dependency.
- **LlamaIndex.** Overlaps heavily with LangChain. Choose-one situation; LangChain wins on LangGraph + LangSmith integration.
- **Haystack.** Solid, but smaller ecosystem for the agentic + eval pieces we need.
- **DSPy for prompt optimization.** Tempting for the self-improvement loop, but we want component-level evolution (chunker, retriever, reranker swaps), not prompt-token optimization. Could be layered in later for tuning prompts inside fixed pipelines.

---

## 4. Repository Structure

```
knowledge-agent/
├── README.md
├── SPEC.md                          # this file
├── pyproject.toml
├── uv.lock
├── .env.example
├── configs/
│   ├── default.yaml
│   ├── components.yaml              # the component registry
│   ├── eval.yaml
│   └── self_improvement.yaml
├── src/
│   ├── harness/
│   │   ├── orchestrator/            # LangGraph graphs
│   │   ├── planning/
│   │   ├── memory/
│   │   ├── compaction/
│   │   ├── subagents/
│   │   ├── sandbox/
│   │   ├── skills/
│   │   ├── context/                 # context packing
│   │   ├── observability/
│   │   ├── permissions/
│   │   ├── budget/
│   │   ├── cache/
│   │   └── citation/
│   ├── knowledge_index/
│   │   ├── ingestion/
│   │   ├── chunking/
│   │   ├── enrichment/
│   │   ├── embedding/
│   │   ├── indexing/
│   │   ├── retrieval/
│   │   │   ├── routers/
│   │   │   ├── retrievers/
│   │   │   ├── query_ops/
│   │   │   ├── fusion/
│   │   │   ├── reranking/
│   │   │   ├── post/
│   │   │   └── iterative/
│   │   └── graph/
│   ├── self_improvement/
│   │   ├── registry/
│   │   ├── ledger/                  # versioned experiment records (JSONL + git)
│   │   ├── evolutionary/
│   │   ├── reviewer/
│   │   ├── pr_gen/
│   │   └── budget_guard/
│   ├── evaluation/
│   │   ├── datasets/
│   │   ├── metrics/
│   │   ├── runners/
│   │   └── e2e/
│   └── common/
│       ├── schemas.py               # Pydantic models shared across modules
│       ├── types.py
│       └── errors.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/
├── experiments/                     # versioned experiment ledger (committed to git)
│   ├── runs/
│   └── lineage.parquet
├── scripts/
│   ├── ingest.py
│   ├── eval_run.py
│   └── self_improve_run.py
└── .github/
    └── workflows/
        ├── ci.yaml                  # ruff + mypy + pytest unit/integration
        └── eval.yaml                # nightly retrieval + e2e eval suite on dev set
```

---

## 5. Core Schemas (`src/common/schemas.py`)

These types are referenced by every module. Build them first.

```python
from pydantic import BaseModel, Field
from typing import Literal, Any
from datetime import datetime
from uuid import UUID

class Source(BaseModel):
    doc_id: str
    chunk_id: str
    parent_id: str | None = None
    title: str | None = None
    url: str | None = None
    span: tuple[int, int] | None = None  # char offsets within parent doc
    metadata: dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    parent_id: str | None = None
    text: str
    context: str | None = None  # contextual retrieval enrichment
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: list[str] = Field(default_factory=list)  # principals allowed to read

class Query(BaseModel):
    raw: str
    rewrites: list[str] = Field(default_factory=list)
    hyde: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    intent: Literal["lookup", "synthesis", "comparison", "relational", "unknown"] = "unknown"
    filters: dict[str, Any] = Field(default_factory=dict)
    user_principals: list[str] = Field(default_factory=list)

class RetrievalCandidate(BaseModel):
    chunk: Chunk
    score: float
    retriever: str  # which retriever produced it
    rank: int

class RetrievalResult(BaseModel):
    candidates: list[RetrievalCandidate]
    query: Query
    trace_id: UUID
    cost: float = 0.0
    latency_ms: float = 0.0

class Citation(BaseModel):
    source: Source
    quote: str | None = None  # short verbatim span if used
    claim_span: tuple[int, int]  # offsets within the generated response

class GenerationResult(BaseModel):
    text: str
    citations: list[Citation]
    trace_id: UUID
    cost: float
    tokens_in: int
    tokens_out: int

class Plan(BaseModel):
    goal: str
    steps: list["PlanStep"]
    status: Literal["draft", "executing", "completed", "failed"]

class PlanStep(BaseModel):
    id: str
    description: str
    tool: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    result: Any = None
```

---

## 6. Harness Modules

### 6.1 Orchestrator (`harness/orchestrator/`) — Phase 1

The agentic loop. LangGraph state machine with these nodes:

```
        ┌──────────┐
        │   plan   │
        └────┬─────┘
             │
        ┌────▼─────┐
        │  route   │  (decide: respond | retrieve | tool | sub-agent)
        └────┬─────┘
             │
   ┌─────────┼─────────┐──────────┐
   ▼         ▼         ▼          ▼
┌──────┐ ┌──────┐ ┌─────────┐ ┌─────────┐
│answer│ │knowl-│ │  tool   │ │sub-agent│
│      │ │ edge │ │         │ │         │
└──┬───┘ └──┬───┘ └────┬────┘ └────┬────┘
   │        │          │            │
   │        └──────────┴────────────┘
   │                   │
   │              ┌────▼────┐
   │              │ observe │  (digest results, update state)
   │              └────┬────┘
   │                   │
   │              ┌────▼────┐
   │              │compact? │  (if context > threshold)
   │              └────┬────┘
   │                   │
   │                   └──────► back to route
   ▼
 ┌────────┐
 │finalize│  (citation enforcement, output formatting)
 └────────┘
```

**Implementation:** Use `langgraph.StateGraph`. State is a `TypedDict` carrying messages, plan, intermediate results, budget remaining, and trace context.

**Interface:**

```python
# harness/orchestrator/graph.py
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

def build_orchestrator(config: OrchestratorConfig) -> StateGraph:
    """Return a compiled LangGraph with checkpointing enabled."""
    ...

class OrchestratorState(TypedDict):
    messages: list[BaseMessage]
    plan: Plan | None
    pending_tools: list[ToolCall]
    retrieval_results: list[RetrievalResult]
    citations: list[Citation]
    budget: BudgetState
    trace_id: UUID
```

**Acceptance:**
- Can answer a question that requires 0, 1, or 2 retrieval calls.
- Spawns a sub-agent with isolated context when `route` selects it.
- Hits compaction when `len(messages_tokens) > config.compaction_threshold`.
- Every output passes through `finalize` which fails if claims lack citations (configurable strictness).

---

### 6.2 Planning (`harness/planning/`) — Phase 1

Two modes, selectable per request:

- **ReAct** for simple queries (no explicit plan; route → act → observe).
- **Plan-and-Execute** for complex queries: LLM emits a todo list, executes serially or parallelizes independent steps.

**Interface:**

```python
class Planner(Protocol):
    async def plan(self, goal: str, context: PlanningContext) -> Plan: ...
    async def adapt(self, plan: Plan, new_observation: Any) -> Plan: ...

class TodoListPlanner(Planner): ...  # Plan-and-Execute
class ReactPlanner(Planner): ...     # null-plan; one step at a time
```

**Acceptance:**
- Plans serialize to/from JSON for inspection.
- Plan adaptation triggers when a step fails or returns unexpected results.
- Independent steps execute concurrently (`asyncio.gather`).

---

### 6.3 Memory (`harness/memory/`) — Phase 2

Three layers:

1. **Working memory** — current turn's scratchpad. In-process dict; cleared per turn.
2. **Session memory** — last N turns + extracted facts. SQLite via LangGraph checkpointer.
3. **Long-term memory** — durable user/project facts. Stored as a vector index. **Use a separate Qdrant collection** (`ka_memory_longterm`), not a payload-filter namespace within the main corpus collection. Reasons: (a) ACL surface area differs — memory is per-user, corpus is shared with principals; (b) reindexing the main corpus shouldn't disturb memory; (c) snapshot/restore cycles run on different cadences. Reuse the same embedder, embedding cache, and retrieval interfaces.

**Interface:**

```python
class Memory(Protocol):
    async def write(self, key: str, value: Any, scope: Literal["working","session","long_term"]): ...
    async def read(self, query: str, scope: Literal["working","session","long_term"], k: int=5) -> list[MemoryItem]: ...
    async def forget(self, predicate: Callable[[MemoryItem], bool]): ...
```

**Long-term memory writes go through an extraction step** — an LLM call that decides what's worth remembering (preferences, facts about the user, recurring entities). Do not naively store everything.

---

### 6.4 Sub-agents (`harness/subagents/`) — Phase 2

Clean-context delegation. Parent passes:

```python
class SubAgentTask(BaseModel):
    task: str
    return_schema: type[BaseModel]  # parent specifies what shape it wants back
    tools: list[str]                # subset of tools available to sub-agent
    budget: BudgetSpec
    max_turns: int = 20
```

Sub-agent receives ONLY this task. No parent message history. Returns a structured result matching `return_schema`.

**Implementation:** Each sub-agent is its own LangGraph instance, instantiated fresh. Spawn via `asyncio.create_task` for parallelism.

**Acceptance:**
- Sub-agent context contains ONLY the task and its own working state.
- Parent receives only the structured result + a trace pointer (for debugging).
- Multiple sub-agents can run in parallel.
- Sub-agent budget is enforced and bubbles up to parent.

---

### 6.5 Compaction (`harness/compaction/`) — Phase 2

Triggered when context approaches model limit. Strategies (configurable):

1. **Hierarchical summarization** — summarize old turns into a single system note.
2. **Selective retention** — keep recent N turns + retrieved chunks; drop tool I/O verbatim, replace with summary.
3. **Off-loading to memory** — extract durable facts to long-term memory before dropping.

**Interface:**

```python
class Compactor(Protocol):
    async def should_compact(self, state: OrchestratorState) -> bool: ...
    async def compact(self, state: OrchestratorState) -> OrchestratorState: ...
```

**Acceptance:**
- Compaction preserves: original goal, current plan, last 3 turns, all citations.
- Compaction drops: raw tool outputs, intermediate retrieval candidates that weren't cited, exploratory branches.
- Round-trips: a compacted state can continue the conversation coherently.

---

### 6.6 Context Engineering / Packing (`harness/context/`) — Phase 1

Distinct from compaction. Decides **what goes into the prompt for the next LLM call**: system prompt, skills, memory hits, retrieved chunks, scratchpad — and in what order.

**Why it matters:** lost-in-the-middle is real. Position and order materially affect generation quality.

**Interface:**

```python
class ContextPacker(Protocol):
    def pack(
        self,
        system: str,
        skills: list[Skill],
        memory_hits: list[MemoryItem],
        retrieval: RetrievalResult | None,
        scratchpad: str,
        messages: list[BaseMessage],
        budget_tokens: int,
    ) -> list[BaseMessage]: ...
```

**Default policy:**
- System prompt (frozen, prompt-cached).
- Skills relevant to current step.
- Retrieved chunks ordered by reranker score, with most-relevant at both top and bottom of the chunk block (combat lost-in-the-middle).
- Recent conversation turns last.

---

### 6.7 Sandbox (`harness/sandbox/`) — Phase 3

Tool execution isolation. Critical for `code_execution`, `web_fetch`, anything that could be hijacked by prompt injection in retrieved content.

**Implementation:**
- Each tool call runs in an ephemeral Docker container with no network by default (network is opt-in per-tool).
- File system mounts limited to a job-scoped temp dir.
- CPU/memory/timeout limits per call.

**Interface:**

```python
class Sandbox(Protocol):
    async def run(self, tool: Tool, args: dict, policy: SandboxPolicy) -> ToolResult: ...

class SandboxPolicy(BaseModel):
    network: Literal["none","allowlist","full"] = "none"
    allowlist: list[str] = []
    cpu_seconds: int = 30
    memory_mb: int = 512
    fs_writable: bool = True
```

---

### 6.8 Skills (`harness/skills/`) — Phase 2

Loadable instruction packs. Each skill is a directory with a `SKILL.md` describing when to use it, plus optional helper scripts.

**Loader:**

```python
class SkillRegistry:
    def discover(self, root: Path) -> list[SkillManifest]: ...
    def select(self, query: Query, available: list[SkillManifest], k: int=2) -> list[Skill]: ...
    def load(self, name: str) -> Skill: ...
```

Selection uses a small classifier (fast LLM call with skill descriptions in context) to pick the relevant skills for the current step.

---

### 6.9 Observability (`harness/observability/`) — Phase 1

Wraps everything in LangSmith traces + OpenTelemetry spans.

**What must be traceable:**
- Every LLM call (model, tokens, cost, latency, cache hit/miss).
- Every retrieval call (query, candidates pre-rerank, candidates post-rerank, chosen, citations).
- Every tool call (sandbox policy, args, result hash).
- Every state transition in the orchestrator graph.

**Interface:**

```python
# Decorator-based for ergonomics
@traced(span_name="retrieval.hybrid")
async def hybrid_retrieve(query: Query) -> RetrievalResult: ...
```

---

### 6.10 Permissions / Approval Gates (`harness/permissions/`) — Phase 2

For destructive or sensitive actions. The orchestrator pauses at a gate node and emits an approval request; the chat layer surfaces it to the user; the user's response unblocks the graph (LangGraph supports this natively via `interrupt`).

**Default gates:**
- Writes to external systems (email send, file write outside sandbox).
- Tool calls with budget > `gate_budget_threshold`.
- Sub-agent spawning beyond `max_concurrent_subagents`.

---

### 6.11 Budget Tracker (`harness/budget/`) — Phase 1

Per-request token/cost budgets, with sub-budgets for sub-agents.

**Interface:**

```python
class BudgetTracker:
    def reserve(self, amount: float) -> BudgetGrant: ...
    def consume(self, grant: BudgetGrant, actual: float) -> None: ...
    def remaining(self) -> float: ...
    def child_budget(self, fraction: float) -> "BudgetTracker": ...  # for sub-agents
```

Orchestrator checks remaining budget at each `route` step; if low, finalizes early with a "ran out of budget" caveat.

---

### 6.12 Cache (`harness/cache/`) — Phase 1

Three tiers:

1. **Prompt cache** — handled by Anthropic API (`cache_control` blocks). Wrap the LLM client to apply caching to system prompt + skills + long retrieved contexts.
2. **Embedding cache** — keyed on `(model, text_hash)`. Backed by Redis or SQLite.
3. **Retrieval cache** — keyed on `(query_hash, index_version, filters_hash)`. Short TTL (e.g., 5 min) to allow index updates to flow through.

---

### 6.13 Citation (`harness/citation/`) — Phase 1

The `finalize` step in the orchestrator graph runs the citation enforcer.

**Approach:** structured generation. The model is constrained to output text in segments, each tagged with a citation ID drawn from the retrieved candidate set. Segments without backing candidates either get flagged or removed (configurable).

**Generation schema** (enforced via Anthropic tool-use / JSON-mode):

```python
class CitedSegment(BaseModel):
    text: str                       # one claim or contiguous span of supported text
    citation_ids: list[str]         # chunk_ids from the candidate set; [] means uncited
    confidence: Literal["high","medium","low"] = "high"

class CitedDraft(BaseModel):
    segments: list[CitedSegment]
    refused: bool = False           # set true if the model declines (e.g., no evidence at all)
    refusal_reason: str | None = None
```

The model is given the candidate set as `[{chunk_id, text}, ...]` and must emit a `CitedDraft`. The enforcer then:
1. Validates every `citation_ids` entry references a candidate that was actually in the set (no hallucinated IDs).
2. Re-flows the segments into final prose, computing `claim_span` offsets for each `Citation`.
3. Applies strictness policy.

**Interface:**

```python
class CitationEnforcer:
    async def enforce(
        self,
        draft: str,
        candidates: list[RetrievalCandidate],
        strictness: Literal["strict","loose","off"] = "strict",
    ) -> GenerationResult: ...
```

In `strict` mode, any unsupported claim either gets removed or returned with an error. In `loose` mode, unsupported claims are tagged as `[uncited]` for the UI to render differently. In `off` mode, the enforcer is bypassed (use only for offline evaluation runs that score citations separately).

---

## 7. Knowledge Index

The main tool. Build this module-by-module — it's where most of the engineering lives.

### 7.1 Ingestion (`knowledge_index/ingestion/`) — Phase 1

**Responsibilities:**
- Parse: PDFs (use `pymupdf4llm` for layout-aware extraction, fallback to `unstructured` for complex docs), DOCX (`python-docx`), HTML (`trafilatura`), Markdown (native), code files (tree-sitter).
- Extract metadata: title, author, date, section headings, source URL, file path.
- Deduplicate: MinHash LSH over normalized text to catch near-duplicates.
- Normalize: Unicode NFC, whitespace collapse, encoding detection.

**Interface:**

```python
class Parser(Protocol):
    async def parse(self, blob: bytes, hint: MimeType) -> ParsedDoc: ...

class ParsedDoc(BaseModel):
    doc_id: str
    text: str
    structure: list[StructureElement]  # headings, tables, code blocks, figures
    metadata: dict[str, Any]
```

**Acceptance:**
- 100 mixed-format documents ingested in <60s on a laptop.
- Tables in PDFs are preserved as markdown tables (not flattened).
- Near-duplicate (>90% Jaccard) flagged but not auto-removed; metadata records the cluster.

---

### 7.2 Chunking (`knowledge_index/chunking/`) — Phase 1

**Implement all of these strategies behind a common interface.** They are registry components for the self-improvement loop.

```python
class Chunker(Protocol):
    name: str
    config: dict[str, Any]
    def chunk(self, doc: ParsedDoc) -> list[Chunk]: ...
```

Strategies:

| Name | Lib | Config |
|---|---|---|
| `recursive` | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `chunk_size`, `chunk_overlap`, `separators` |
| `markdown_header` | `langchain_text_splitters.MarkdownHeaderTextSplitter` then recursive | header levels |
| `semantic` | `langchain_experimental.text_splitter.SemanticChunker` | breakpoint type, threshold |
| `sentence_window` | Custom: 1-sentence chunks + window for context | window size |
| `late_chunking` | Custom: embed full doc, pool by sentence boundaries | pool method |
| `propositional` | LLM-based: decompose into atomic propositions | LLM model |

**Default:** `recursive` with `chunk_size=500, chunk_overlap=75` on markdown, switch to `markdown_header` first for structured docs.

---

### 7.3 Enrichment (`knowledge_index/enrichment/`) — Phase 1

**Contextual retrieval implementation.** For each chunk, prepend a 1–3 sentence chunk-specific context generated by an LLM with the full document available (cached via Anthropic prompt caching to make this affordable at scale).

```python
class Enricher(Protocol):
    name: str
    async def enrich(self, doc: ParsedDoc, chunks: list[Chunk]) -> list[Chunk]: ...

class ContextualEnricher(Enricher):
    """Anthropic-style contextual retrieval."""
    async def enrich(self, doc, chunks):
        # Cache the document in the prompt; iterate chunks; LLM emits per-chunk context
        ...
```

**Prompt template:**

```
<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short (max 100 tokens) succinct context to situate this chunk within
the overall document, for the purposes of improving search retrieval of the
chunk. Answer only with the succinct context and nothing else.
```

Other enrichers to implement:
- `TitleEnricher` — prepend doc title + section path.
- `SummaryEnricher` — prepend a one-sentence summary of the chunk.
- `Null` — no enrichment (baseline).

---

### 7.4 Embedding (`knowledge_index/embedding/`) — Phase 1

```python
class Embedder(Protocol):
    name: str
    dim: int
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
```

Implementations: `voyage-3-large`, `text-embedding-3-large`, `bge-large-en-v1.5` (local fallback). All wrapped through the cache.

---

### 7.5 Indexing (`knowledge_index/indexing/`) — Phase 1

**Qdrant** as the primary store. Hybrid collection: dense vectors + BM25 sparse vectors in the same collection.

```python
class Index(Protocol):
    async def upsert(self, chunks: list[Chunk]): ...
    async def delete(self, chunk_ids: list[str]): ...
    async def search_dense(self, vec: list[float], k: int, filters: dict) -> list[RetrievalCandidate]: ...
    async def search_sparse(self, query: str, k: int, filters: dict) -> list[RetrievalCandidate]: ...
    async def snapshot(self) -> SnapshotRef: ...     # for index versioning
    async def restore(self, ref: SnapshotRef): ...
```

**ACL enforcement happens here** via payload filters, not after retrieval. Filter on `acl` field; user's principals must intersect.

---

### 7.6 Retrieval Pipeline

The retrieval pipeline is itself a small graph. Each stage is a registry component.

```
Query
  │
  ▼
┌─────────────┐
│  Router     │ ─► chooses retrieval strategy
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Query Ops   │ ─► rewrite, HyDE, decompose
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Retrievers  │ ─► run in parallel: dense, BM25, graph
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Fusion (RRF)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Reranker    │ ─► cross-encoder
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Post-proc   │ ─► MMR, parent expansion, span extraction
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Packer      │ ─► order for prompt
└─────────────┘
       │
       ▼  (if Iterative router decides)
   Multi-hop ─► loop back with new query informed by results
```

#### 7.6.1 Router (`retrieval/routers/`) — Phase 2

```python
class QueryRouter(Protocol):
    async def route(self, query: Query) -> RouteDecision: ...

class RouteDecision(BaseModel):
    strategy: Literal["naive","hybrid","graph","iterative"]
    intent: Literal["lookup","synthesis","comparison","relational"]
    expected_complexity: Literal["low","medium","high"]
    filters: dict[str, Any]
```

Implementation: small classifier (Haiku-class model with few-shot prompt) returns the strategy. Cache aggressively per query hash.

#### 7.6.2 Query Operations (`retrieval/query_ops/`) — Phase 1

```python
class QueryOp(Protocol):
    name: str
    async def transform(self, query: Query) -> Query: ...

class Rewriter(QueryOp): ...        # LLM-based query rewrite
class HyDEExpander(QueryOp): ...    # generate hypothetical answer doc, embed it
class Decomposer(QueryOp): ...      # break multi-part query into sub-queries
class Stepback(QueryOp): ...        # generate broader version of query
```

Composable as a list: `[Rewriter(), HyDEExpander()]` applied in order.

#### 7.6.3 Retrievers (`retrieval/retrievers/`) — Phase 1

```python
class Retriever(Protocol):
    name: str
    async def retrieve(self, query: Query, k: int) -> list[RetrievalCandidate]: ...

class DenseRetriever(Retriever): ...
class SparseBM25Retriever(Retriever): ...
class GraphRetriever(Retriever): ...     # multi-hop traversal on KG
class ParentChildRetriever(Retriever): ... # retrieve child, return parent
```

Run multiple in parallel via `asyncio.gather`, then fuse.

#### 7.6.4 Fusion (`retrieval/fusion/`) — Phase 1

```python
class Fuser(Protocol):
    async def fuse(self, results: list[list[RetrievalCandidate]]) -> list[RetrievalCandidate]: ...

class RRFFuser(Fuser):
    """Reciprocal Rank Fusion. k=60 default."""

class WeightedFuser(Fuser):
    """Score-normalized weighted sum. Weights configurable per retriever."""
```

LangChain's `EnsembleRetriever` already implements RRF; wrap or replace.

#### 7.6.5 Reranking (`retrieval/reranking/`) — Phase 1

```python
class Reranker(Protocol):
    name: str
    async def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]: ...

class CohereReranker(Reranker): ...
class VoyageReranker(Reranker): ...
class LLMReranker(Reranker): ...    # slower, sometimes better
class NullReranker(Reranker): ...   # baseline
```

#### 7.6.6 Post-processing (`retrieval/post/`) — Phase 1

```python
class PostProcessor(Protocol):
    async def process(self, query: Query, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]: ...

class MMRDiversifier(PostProcessor): ...
class ParentExpander(PostProcessor): ...        # swap chunk for parent
class SpanExtractor(PostProcessor): ...         # find exact relevant span within chunk
class DeduplicatorPostProcessor(PostProcessor): ...
```

Composable list.

#### 7.6.7 Iterative / Multi-hop (`retrieval/iterative/`) — Phase 3

The agentic retrieval loop. After the first retrieval round, an LLM examines results and decides:
- Is the answer here? → terminate.
- What's missing? → emit follow-up queries.
- Re-run retrieval with new queries, accumulate evidence.

```python
class IterativeRetriever(Retriever):
    async def retrieve(self, query: Query, k: int) -> list[RetrievalCandidate]:
        evidence = []
        for hop in range(self.max_hops):
            results = await self.inner.retrieve(query, k)
            evidence.extend(results)
            decision = await self.judge(query, evidence)
            if decision.done:
                break
            query = decision.next_query
        return self.deduplicate_and_rank(evidence)
```

Budget-aware: hops cost tokens; tracker enforces ceiling.

---

### 7.7 Graph Layer (`knowledge_index/graph/`) — Phase 3

For GraphRAG route. Build during ingestion:
- Extract entities (NER + LLM-based extraction for domain terms).
- Extract relations (LLM-based triplet extraction with a constrained schema).
- Store in Neo4j (or `networkx` for dev).

```python
class GraphBuilder:
    async def build(self, docs: list[ParsedDoc]) -> None: ...

class GraphRetriever(Retriever):
    async def retrieve(self, query: Query, k: int) -> list[RetrievalCandidate]:
        entities = await self.extract_entities(query.raw)
        subgraph = await self.traverse(entities, depth=2)
        # Surface chunks linked to subgraph nodes
        ...
```

---

## 8. Self-Improvement System

### 8.1 Component Registry (`self_improvement/registry/`) — Phase 2

The bounded search space. Every component declares itself with:

```yaml
# configs/components.yaml
chunkers:
  - name: recursive
    class: knowledge_index.chunking.RecursiveChunker
    params:
      chunk_size: {type: int, range: [200, 1500], default: 500}
      chunk_overlap: {type: int, range: [0, 200], default: 75}
  - name: semantic
    class: knowledge_index.chunking.SemanticChunker
    params:
      threshold: {type: float, range: [0.5, 0.95], default: 0.75}

enrichers:
  - name: null
  - name: contextual
    params:
      max_context_tokens: {type: int, range: [30, 150], default: 80}
  - name: title

retrievers:
  - name: dense
  - name: sparse_bm25
  - name: hybrid_rrf
    params:
      rrf_k: {type: int, range: [10, 200], default: 60}

rerankers:
  - name: null
  - name: cohere_rerank_3
  - name: voyage_rerank_2
  - name: llm_rerank
    params:
      model: {type: enum, values: [haiku, sonnet], default: haiku}

post_processors:
  - name: mmr
    params:
      lambda: {type: float, range: [0.0, 1.0], default: 0.5}
  - name: parent_expander
  - name: span_extractor

query_ops:
  - name: rewrite
  - name: hyde
  - name: decompose
  - name: stepback
```

**The registry is the ONLY place the self-improvement loop can pull components from.** New techniques enter via human PRs to this file.

### 8.2 Evolutionary Loop (`self_improvement/evolutionary/`) — Phase 4

AlphaEvolve pattern. Each generation:

1. **Sample candidates** from current population. A *candidate* is a full pipeline config: `(chunker + params, enricher + params, retrievers, fusion, reranker, post_processors, query_ops)`.
2. **Mutate** — change one component or one parameter (small steps).
3. **Crossover** — combine two parents' configs at component boundaries.
4. **Evaluate** on the held-out eval set (see §9).
5. **Adversarial review** (§8.3).
6. **Select** — top-k by composite score advance to next generation.

```python
class EvolutionaryLoop:
    def __init__(self, registry, evaluator, reviewer, budget):
        ...

    async def run(self, generations: int, population_size: int) -> EvolutionReport:
        population = self.seed_population()
        for gen in range(generations):
            offspring = self.mutate_and_cross(population)
            evaluated = await self.evaluator.evaluate_batch(offspring)
            reviewed = await self.reviewer.review_batch(evaluated)
            population = self.select(population + reviewed, population_size)
            if self.budget.exhausted():
                break
        return self.report(population)
```

**Critical constraints:**
- Hard budget cap (compute hours, $).
- Hold-out test set the loop NEVER sees; results checked only at the very end before PR.
- Improvements must clear a delta threshold (e.g., +2% nDCG@10) to qualify.
- Top candidates must show improvement on the rotating eval set too (Goodhart guard).

#### 8.2.1 Experiment ledger (versioned, replayable, git-backed)

Every evolutionary run produces atomic experiment records that must be replayable, auditable, and survivable across crashes. Inspired by the "work-as-primitive with git-versioned state" pattern from MEOW/Beads, without the framework dependency.

Each experiment is an atomic record stored as JSONL under `experiments/` in the repo, with one file per generation. Git is the version control. The schema:

```python
class Experiment(BaseModel):
    experiment_id: str            # uuid7 — time-orderable
    parent_ids: list[str]         # lineage: which experiments this descends from (for crossover, both parents)
    generation: int
    run_id: str                   # groups experiments from one evolutionary run
    config: PipelineConfig        # the full candidate config — chunker, enricher, retrievers, etc.
    config_hash: str              # content-addressable; identical configs share results
    mutation: MutationRecord | None  # what changed vs parent (component swap or param delta)
    status: Literal["pending","running","evaluated","reviewed","accepted","rejected","failed"]
    eval_results: dict[str, MetricResult] | None  # keyed by dataset name
    reviewer_verdict: ReviewerVerdict | None
    cost_usd: float
    compute_seconds: float
    created_at: datetime
    completed_at: datetime | None
    trace_ids: list[UUID]         # LangSmith pointers
    artifacts: dict[str, str]     # paths to serialized indexes, eval outputs, etc.

class MutationRecord(BaseModel):
    type: Literal["mutate","crossover","seed"]
    component: str                # e.g., "chunker"
    change: dict[str, Any]        # before/after diff
```

**Storage layout:**

```
experiments/
├── runs/
│   └── {run_id}/
│       ├── manifest.yaml         # run config: budget, generations, population size, dataset refs
│       ├── gen-001.jsonl         # one Experiment per line
│       ├── gen-002.jsonl
│       └── ...
└── lineage.parquet               # denormalized lineage graph for fast queries
```

**Why JSONL + git instead of a database:**
- Append-only by construction; no concurrent-write coordination.
- `git log` is the audit trail for free.
- Diffable, reviewable in PRs.
- Survives crashes — incomplete experiments are visible as `status: running` and can be resumed or garbage-collected.
- Cheap to ship the whole experiment history with the repo for reproducibility.

**Interface:**

```python
class ExperimentLedger(Protocol):
    async def append(self, exp: Experiment) -> None: ...
    async def update_status(self, experiment_id: str, status: str, **fields) -> None: ...
    async def get(self, experiment_id: str) -> Experiment: ...
    async def lineage(self, experiment_id: str) -> list[Experiment]: ...
    async def query(self, predicate: Callable[[Experiment], bool]) -> list[Experiment]: ...
    async def replay(self, experiment_id: str) -> EvalReport: ...  # re-runs the eval from stored config
```

**Replayability requirement:** given an `experiment_id`, the system must be able to re-execute the experiment end-to-end from its `config` and produce results within noise band of the original. This is the test that the experiment record is complete.

The PR generator (§8.4) references experiment IDs in its evidence package. The adversarial reviewer (§8.3) can query the ledger to compare a candidate against its lineage.

### 8.3 Adversarial Reviewer (`self_improvement/reviewer/`) — Phase 4

A separate LLM-driven step whose explicit job is to find reasons a result is invalid. Addresses the "overexcitement" failure mode documented in autonomous research agents.

Reviewer checks:
- Was the eval set leakage-free? (No overlap with training/seed configs.)
- Did the candidate improve only on a narrow query slice?
- Are improvements within noise band of seed variance?
- Did latency or cost regress beyond threshold?

Reviewer produces a verdict + a structured critique. The verdict (`accept | reject | needs_more_evidence`) gates PR creation.

### 8.4 PR Generation (`self_improvement/pr_gen/`) — Phase 4

For accepted candidates:
- Open a git branch.
- Write the config diff (`configs/default.yaml` update).
- Generate a PR description that includes: eval metrics before/after, candidate lineage, reviewer report, link to LangSmith trace of evaluation run, link to held-out test results.
- Open the PR via GitHub API.
- **A human reviews and merges.** No auto-merge.

### 8.5 Budget Guard (`self_improvement/budget_guard/`) — Phase 4

Hard ceilings:
- Max generations per run.
- Max compute-hours per generation.
- Max $ per run.
- Daily $ ceiling across all runs.

Kill switch if any ceiling breached.

---

## 9. Evaluation Framework

### 9.1 Datasets (`evaluation/datasets/`) — Phase 2

Three datasets:

1. **Dev set** — seen freely by the self-improvement loop during evolution. ~500 queries.
2. **Rotating eval set** — shown to the loop only at generation boundaries, rotated quarterly. ~500 queries.
3. **Frozen test set** — NEVER shown to the loop. Used only for final pre-PR verification and manual audits. ~1000 queries.

Each query has:
```python
class GoldQuery(BaseModel):
    query_id: str
    query: str
    relevant_chunk_ids: list[str]
    relevant_doc_ids: list[str]
    expected_answer: str | None  # for end-to-end eval
    intent: str
    difficulty: Literal["easy","medium","hard"]
    notes: str = ""
```

Bootstrap with LLM-generated queries (with human review) over the indexed corpus.

### 9.2 Metrics (`evaluation/metrics/`) — Phase 2

**Retrieval metrics** (per query, then aggregated):
- Recall@k (k=5, 10, 20).
- Precision@k.
- nDCG@k.
- MRR (Mean Reciprocal Rank).
- Hit rate.

**End-to-end metrics:**
- Answer faithfulness (ragas).
- Answer relevance (ragas).
- Citation precision (every claim has a backing chunk that actually supports it — uses an LLM judge).
- Citation recall (every supportable claim has a citation).

**Operational metrics:**
- Latency (p50, p95).
- Cost per query.
- Token efficiency (useful tokens / total tokens).

### 9.3 Runners (`evaluation/runners/`) — Phase 2

```python
class EvalRunner:
    async def run(
        self,
        pipeline_config: PipelineConfig,
        dataset: Dataset,
        metrics: list[Metric],
    ) -> EvalReport: ...
```

LangSmith integration: every run logs as an experiment with config + results, comparable across runs.

---

## 10. Implementation Phases

Build in this order. Each phase is shippable and adds capability.

### Phase 1 — Foundation (week 1–2)
**Goal:** End-to-end RAG with hybrid retrieval, contextual chunking, reranking, and citations.

- Common schemas, config plumbing, logging.
- Ingestion: PDF + markdown + HTML.
- Chunking: recursive + markdown_header.
- Enrichment: contextual + null.
- Embedding: voyage-3-large with cache.
- Indexing: Qdrant hybrid collection.
- Retrieval: dense + BM25 + RRF + Cohere rerank.
- Post: MMR + parent expander.
- Query ops: rewriter.
- Orchestrator: minimal graph (plan → retrieve → answer → finalize with citations).
- Budget tracker, cache, observability, citation enforcer.

**Acceptance:** can ingest 1000 docs, answer 50 queries with citations, p50 < 4s end-to-end.

### Phase 2 — Agentic Loop (week 3–4)
**Goal:** Planning, sub-agents, memory, full orchestrator.

- Plan-and-execute planner.
- Sub-agents with isolated context.
- Memory layers (working, session, long-term).
- Compaction.
- Permissions gates.
- Skills loader.
- Query router + intent classification.
- Evaluation framework (dev + frozen sets, retrieval + e2e metrics).

**Acceptance:** can handle a complex query requiring planning + 2 sub-agents + 3 retrieval calls. Eval suite runs in CI.

### Phase 3 — Advanced Retrieval (week 5–6)
**Goal:** Iterative retrieval, GraphRAG, full chunker zoo.

- Iterative / multi-hop retriever.
- All chunkers: semantic, sentence_window, late_chunking, propositional.
- All enrichers, all query ops, all post-processors.
- Graph layer + GraphRetriever.
- Sandbox for tool execution.

**Acceptance:** GraphRAG route demonstrably better than vector route on relational queries (test slice). Iterative retrieval improves recall@5 on hard queries by ≥10%.

### Phase 4 — Self-Improvement (week 7–8)
**Goal:** Evolutionary loop in production, gated by adversarial review and human PR review.

- Component registry.
- Evolutionary loop with mutation + crossover.
- Adversarial reviewer.
- PR generator with full evidence package.
- Budget guard.

**Acceptance:** Run the loop end-to-end on the dev set; it surfaces ≥1 improvement that holds up on the rotating set and clears reviewer; opens a PR; human merges.

---

## 10.5 Secrets & Config Hygiene

API keys, dataset paths, and infra endpoints are runtime-only. Conventions:

- All secrets read from `os.environ` via a typed `Settings` model (`pydantic-settings`). No literal API keys anywhere in the repo.
- `.env.example` lists every required env var with placeholder values, grouped by service:
  - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (fallback)
  - `VOYAGE_API_KEY`, `COHERE_API_KEY`
  - `QDRANT_URL`, `QDRANT_API_KEY` (or `QDRANT_HOST`/`QDRANT_PORT` for self-hosted)
  - `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`
  - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (Phase 3)
- `.gitignore` excludes: `.env`, `.env.*`, `experiments/runs/*/artifacts/`, `*.sqlite`, `*.duckdb`, `.langsmith_cache/`.
- CI uses GitHub Actions secrets, not committed values.
- `pre-commit` hooks run `gitleaks` (or `trufflehog`) on every commit; CI runs the same scan on PRs.
- Eval datasets that contain real user content are gitignored under `evaluation/datasets/private/`; only synthetic / public datasets ship in the repo.

The `Settings` model:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    openai_api_key: str | None = None
    voyage_api_key: str
    cohere_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str = "knowledge-agent"
```

Instantiated once at process start; passed via dependency injection. Never re-read inside hot paths.

---

## 11. Acceptance Criteria (Whole-System)

The system is "done" when all of the following hold:

1. End-to-end queries return cited answers with p50 latency < 5s, p95 < 15s.
2. Citation precision > 0.9, citation recall > 0.85 on the frozen test set.
3. Retrieval recall@10 > 0.85 on frozen test set across difficulty levels.
4. Sub-agents demonstrably parallelize: a query needing 3 independent sub-tasks completes in roughly the time of the slowest, not the sum.
5. The self-improvement loop has produced at least one merged PR that improved a real metric on the frozen test set by a statistically significant margin.
6. No PII / sensitive data leaks across ACL boundaries in a red-team test suite.
7. Every claim in every output is traceable to a chunk in LangSmith.
8. Total cost per query (LLM + embedding + reranking) is reported per request.

---

## 12. Open Questions for Implementer

These are decisions to make during build:

- **Vector DB:** Qdrant chosen, but Weaviate or Pinecone also work. Decide based on ops familiarity.
- **Graph DB:** Neo4j for prod is right, but consider Memgraph or even DuckDB-with-pg_graph for simpler ops.
- **Adversarial reviewer LLM:** Use the same family as the main pipeline (Claude) or a different one (e.g., GPT-5) to reduce common-mode bias? Start same-family; consider mixing later.
- **Embedding model:** Run a one-time bake-off on the dev set before locking in.
- **Should the orchestrator graph be one big graph or composed sub-graphs?** Start composed; LangGraph supports nesting cleanly.

---

## 13. Anti-patterns to Avoid

- Hard-coding component choices outside the registry.
- Letting sub-agents inherit the parent's full message history.
- Using the rotating or frozen eval sets during evolutionary search.
- Skipping the citation enforcer because "the model usually grounds correctly."
- Reading content from retrieved chunks as if it were trusted instructions (prompt injection surface).
- Auto-merging self-improvement PRs.
- Compacting away tool I/O without first extracting durable facts to long-term memory.
- Running tools without sandbox in production "because dev is fine."

---

## 14. Quick-Start Commands (for future README)

```bash
uv sync
cp .env.example .env  # fill in API keys

# ingest
uv run scripts/ingest.py --src ./docs --collection main

# query
uv run python -c "from harness import answer; print(answer('what is X?'))"

# eval
uv run scripts/eval_run.py --dataset frozen --pipeline configs/default.yaml

# self-improvement run
uv run scripts/self_improve_run.py --generations 5 --population 8 --budget-usd 50
```

---

## 15. Direct Corpus Interaction (DCI) — Phase 5

**Motivation.** 2026 research (Sun et al., "Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction," arxiv 2605.05242; LlamaIndex 2026 fs-explorer benchmarks; A-RAG; Anthropic Claude Code's published shift from vector RAG to grep-style search) shows agents using filesystem-style tools directly over the raw corpus often beat vector RAG on exact-lexical, multi-hop, and code-like queries (reported +11% to +30.7% on 13 benchmarks). Vector hybrid still wins on latency at >1000 docs and paraphrastic semantic matching. We add DCI as a first-class, routed strategy alongside the existing pipeline.

### 15.1 Tools (`src/knowledge_index/dci/`)
Each tool is registered as a Skill (§6.5) and executes inside the Phase 3N sandbox (§6.7). All return values carry citation metadata. All inputs are ACL-filtered against the caller's slice (§11 #6).

- `corpus_grep(pattern, *, glob='**/*', regex=True, max_hits=50, context_lines=2) -> list[GrepHit]` — ACL-filtered regex over raw doc text. GrepHit carries (doc_id, line_no, snippet, ±context, citation).
- `corpus_glob(pattern, *, types=None, limit=200) -> list[DocRef]` — path-pattern listing within the ACL slice.
- `corpus_ls(path='/') -> DirectoryListing` — browse the logical tree (collection→source→doc).
- `corpus_read(doc_id, *, start_line=1, end_line=None, max_bytes=50000) -> DocSlice` — full-or-windowed doc read with citation.
- `corpus_describe(doc_id) -> DocMetadata` — title/source/authors/length/ACL tags/ingestion time.
- `corpus_neighbors(chunk_id, *, hops=1) -> list[ChunkRef]` — KG walk (Phase 3M GraphRetriever) from a chunk.

### 15.2 Routing (extends §7 QueryRouter, Phase 2G)
Add `strategy='dci'` and two chained modes: `dci_then_vector`, `vector_then_dci`. Router heuristics:
- Quoted exact phrases, identifier-like tokens, code-style queries → `dci` (grep first).
- Multi-hop with named-entity bridging → `dci_then_vector` (grep candidates, then expand via retrieval).
- Paraphrastic / large-corpus realtime → existing `hybrid`.

### 15.3 Orchestrator wiring (extends §6.1 Phase 2I)
New node `dci_tool` between `route` and `observe`. Plan steps target DCI tools instead of (or in addition to) `retrieve`. Results flow through the existing citation enforcer and budget tracker; budget tracker counts DCI tool-token usage.

### 15.4 Sandboxing & safety (§6.7, §13)
- Tools run inside the Phase 3N sandbox with read-only mounts of the ACL-permitted corpus slice; deny network, deny writes, time/memory caps.
- Retrieved raw text wrapped in `<corpus_content>` and never executed as instructions (§13 anti-pattern: "Reading content from retrieved chunks as if it were trusted instructions").
- `max_hits` and `max_bytes` enforce token caps; long files window-iterated.
- The Gap G2 ACL/PII red-team suite (ka-wig) is extended with DCI-specific prompt-injection and ACL-bypass probes.

### 15.5 Acceptance criteria
1. On a lexical-bridge eval slice (queries with exact quoted phrases or unique identifiers), `dci` route beats the vector hybrid route by ≥10% recall@5.
2. On a paraphrastic slice, vector hybrid ties or beats `dci` (within noise) — demonstrates routing correctness.
3. On a multi-hop slice (BrowseComp-Plus-style or local synthetic), `dci_then_vector` chained mode beats either alone.
4. All DCI tool calls run in the sandbox with no escapes; red-team suite passes.
5. Citations from DCI tools attribute correctly to source docs (precision >0.95 on the citation audit slice).
6. p50 latency for `dci` route on n=1000 corpus < 8s; chained mode < 12s.

### 15.6 Self-improvement integration (§8.1)
Add `dci_tools` and routing-weight parameters (per-query-class DCI vs vector mix) to `configs/components.yaml` as evolvable. The Phase 4 evolutionary loop can tune the DCI/vector mix per workload.

---

**End of spec.** Hand to Claude Code. Build Phase 1 first.
