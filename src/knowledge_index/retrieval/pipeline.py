"""Hybrid retrieval pipeline assembly (SPEC §7.6).

Composes the retrieval graph end to end::

    QueryOps → parallel Retrievers → Fusion (RRF) → Rerank (Cohere) → PostProc → top-k

Every stage is a registry component injected at construction, so the
self-improvement loop can swap any one (e.g. RRF→Weighted, Cohere→Null) without
touching this file. The single entry point, :meth:`retrieve`, returns a
:class:`RetrievalResult` carrying the candidates plus trace/latency telemetry.

Stages are individually ``@traced``; the pipeline additionally logs candidate-set
sizes at the pre-rerank, post-rerank, and final checkpoints so LangSmith captures
how the set narrows.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid4

from common.schemas import Query, RetrievalCandidate, RetrievalResult
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from knowledge_index.retrieval.fusion.base import Fuser
from knowledge_index.retrieval.fusion.rrf import RRFFuser
from knowledge_index.retrieval.post.base import PostProcessor
from knowledge_index.retrieval.post.mmr import MMRDiversifier
from knowledge_index.retrieval.post.parent import FetchParentFn, ParentExpander
from knowledge_index.retrieval.query_ops.base import QueryOp, apply_query_ops
from knowledge_index.retrieval.reranking.base import Reranker
from knowledge_index.retrieval.reranking.cohere import CohereReranker, SupportsCohereRerank
from knowledge_index.retrieval.reranking.null import NullReranker
from knowledge_index.retrieval.retrievers.base import (
    Retriever,
    SupportsEmbedQuery,
    SupportsSearch,
    gather_retrievers,
)
from knowledge_index.retrieval.retrievers.dense import DenseRetriever
from knowledge_index.retrieval.retrievers.graph import GraphRetriever
from knowledge_index.retrieval.retrievers.sparse import SparseBM25Retriever

if TYPE_CHECKING:
    from knowledge_index.graph.base import EntityExtractor, GraphStore

_log = get_logger("knowledge_index.retrieval.pipeline")

# First-stage fan-out: each retriever fetches more than the final k so fusion and
# reranking have headroom. Tuned conservatively; override per request.
DEFAULT_RETRIEVE_MULTIPLIER = 4
DEFAULT_MIN_RETRIEVE = 20


class HybridPipeline:
    """End-to-end hybrid retrieval pipeline (SPEC §7.6)."""

    def __init__(
        self,
        retrievers: list[Retriever],
        reranker: Reranker,
        *,
        fuser: Fuser | None = None,
        query_ops: list[QueryOp] | None = None,
        post_processors: list[PostProcessor] | None = None,
        retrieve_k: int | None = None,
        rerank_k: int | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridPipeline requires at least one retriever")
        self._retrievers = retrievers
        self._reranker = reranker
        self._fuser = fuser or RRFFuser()
        self._query_ops = query_ops or []
        self._post_processors = post_processors or []
        self._retrieve_k = retrieve_k
        self._rerank_k = rerank_k

    def _first_stage_k(self, k: int) -> int:
        return self._retrieve_k or max(k * DEFAULT_RETRIEVE_MULTIPLIER, DEFAULT_MIN_RETRIEVE)

    def _rerank_top(self, k: int) -> int:
        # Rerank to at least k so post-processors that trim (MMR) still leave k.
        return max(self._rerank_k or k * 2, k)

    @traced(span_name="retrieval.pipeline")
    async def retrieve(self, query: Query, k: int) -> RetrievalResult:
        if k <= 0:
            raise ValueError("k must be positive")
        start = time.perf_counter()
        trace_id = uuid4()

        transformed = await apply_query_ops(self._query_ops, query)

        per_retriever = await gather_retrievers(
            self._retrievers, transformed, self._first_stage_k(k)
        )
        fused = await self._fuser.fuse(per_retriever)
        _log.info(
            "retrieval.pre_rerank",
            trace_id=str(trace_id),
            retrievers=len(self._retrievers),
            fused=len(fused),
        )

        reranked = await self._reranker.rerank(query.raw, fused, self._rerank_top(k))
        _log.info("retrieval.post_rerank", trace_id=str(trace_id), reranked=len(reranked))

        candidates: list[RetrievalCandidate] = reranked
        for post in self._post_processors:
            candidates = await post.process(transformed, candidates)

        final = candidates[:k]
        latency_ms = (time.perf_counter() - start) * 1000.0
        _log.info(
            "retrieval.final",
            trace_id=str(trace_id),
            final=len(final),
            latency_ms=round(latency_ms, 2),
        )

        return RetrievalResult(
            candidates=final,
            query=transformed,
            trace_id=trace_id,
            cost=0.0,  # LLM/embedding cost is recorded per-call in observability (SPEC §6.9)
            latency_ms=latency_ms,
        )


def build_default_pipeline(
    index: SupportsSearch,
    embedder: SupportsEmbedQuery,
    *,
    query_ops: list[QueryOp] | None = None,
    fetch_parent: FetchParentFn | None = None,
    cohere_client: SupportsCohereRerank | None = None,
    mmr_lambda: float = 0.5,
) -> HybridPipeline:
    """Wire the canonical Phase-1 hybrid pipeline (SPEC §7.6 / epic ka-2ap).

    QueryOps → [DenseRetriever ‖ SparseBM25Retriever] → RRF → Cohere rerank →
    [MMR, ParentExpander]. Convoy B supplies ``index`` and ``embedder``;
    ``fetch_parent`` (an index lookup) enables parent expansion when provided.

    Components are still individually swappable — this is just the default wiring
    the orchestrator (Convoy D) and the eval baseline start from.
    """
    post: list[PostProcessor] = [MMRDiversifier(embedder.embed_query, lambda_=mmr_lambda)]
    if fetch_parent is not None:
        post.append(ParentExpander(fetch_parent))

    return HybridPipeline(
        retrievers=[DenseRetriever(index, embedder), SparseBM25Retriever(index)],
        reranker=CohereReranker(client=cohere_client),
        fuser=RRFFuser(),
        query_ops=query_ops or [],
        post_processors=post,
    )


def build_graph_variant(
    store: "GraphStore",
    extractor: "EntityExtractor",
    *,
    depth: int = 2,
    reranker: Reranker | None = None,
    post_processors: list[PostProcessor] | None = None,
) -> HybridPipeline:
    """Wire the GraphRAG strategy variant (SPEC §7.7 / strategy='graph').

    A single :class:`GraphRetriever` over the KG, wrapped in the standard pipeline
    so it returns a :class:`RetrievalResult` and drops straight into
    :class:`~knowledge_index.retrieval.routers.pipeline.RouterPipeline`'s
    ``variants={"graph": ...}``. Fusion is a no-op (one retriever) and the default
    reranker is :class:`NullReranker` — graph proximity is already the score; pass
    a cross-encoder reranker to re-order the surfaced chunks by query relevance.
    """
    return HybridPipeline(
        retrievers=[GraphRetriever(store, extractor, depth=depth)],
        reranker=reranker or NullReranker(),
        fuser=RRFFuser(),
        post_processors=post_processors or [],
    )


__all__ = [
    "DEFAULT_RETRIEVE_MULTIPLIER",
    "DEFAULT_MIN_RETRIEVE",
    "HybridPipeline",
    "build_default_pipeline",
    "build_graph_variant",
]
