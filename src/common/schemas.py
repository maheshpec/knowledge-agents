"""Core Pydantic models shared across every module (SPEC §5).

These types are referenced everywhere; they are the stable contract between
ingestion, retrieval, orchestration, and evaluation. Build them first.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Provenance pointer for a chunk or citation."""

    doc_id: str
    chunk_id: str
    parent_id: str | None = None
    title: str | None = None
    url: str | None = None
    span: tuple[int, int] | None = None  # char offsets within parent doc
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit of text with optional contextual enrichment and ACL."""

    chunk_id: str
    doc_id: str
    parent_id: str | None = None
    text: str
    context: str | None = None  # contextual retrieval enrichment (SPEC §7.3)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: list[str] = Field(default_factory=list)  # principals allowed to read


class Query(BaseModel):
    """A user query plus all derived transformations (rewrites, HyDE, sub-queries)."""

    raw: str
    rewrites: list[str] = Field(default_factory=list)
    hyde: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    intent: Literal["lookup", "synthesis", "comparison", "relational", "unknown"] = "unknown"
    filters: dict[str, Any] = Field(default_factory=dict)
    user_principals: list[str] = Field(default_factory=list)


class RetrievalCandidate(BaseModel):
    """A single retrieved chunk with its score, source retriever, and rank."""

    chunk: Chunk
    score: float
    retriever: str  # which retriever produced it
    rank: int


class RetrievalResult(BaseModel):
    """The output of a retrieval call: ranked candidates plus cost/latency telemetry."""

    candidates: list[RetrievalCandidate]
    query: Query
    trace_id: UUID
    cost: float = 0.0
    latency_ms: float = 0.0


class Citation(BaseModel):
    """A claim in the generated response backed by a source span."""

    source: Source
    quote: str | None = None  # short verbatim span if used
    claim_span: tuple[int, int]  # offsets within the generated response


class GenerationResult(BaseModel):
    """Final generated answer with citations and cost accounting."""

    text: str
    citations: list[Citation]
    trace_id: UUID
    cost: float
    tokens_in: int
    tokens_out: int


class PlanStep(BaseModel):
    """A single step within a plan, with dependencies and execution status."""

    id: str
    description: str
    tool: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    result: Any = None


class Plan(BaseModel):
    """An ordered set of steps toward a goal, with overall execution status."""

    goal: str
    steps: list[PlanStep]
    status: Literal["draft", "executing", "completed", "failed"] = "draft"


# Resolve the forward reference Plan -> PlanStep declared in SPEC §5.
Plan.model_rebuild()


__all__ = [
    "Source",
    "Chunk",
    "Query",
    "RetrievalCandidate",
    "RetrievalResult",
    "Citation",
    "GenerationResult",
    "Plan",
    "PlanStep",
]
