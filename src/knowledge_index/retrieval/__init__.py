"""Retrieval pipeline (SPEC §7.6): query ops, retrievers, fusion, rerank, post-proc.

Re-exports the public surface so callers can ``from knowledge_index.retrieval
import HybridPipeline, RRFFuser, CohereReranker`` without reaching into submodules.
"""

from knowledge_index.retrieval.fusion import (
    DEFAULT_RRF_K,
    Fuser,
    RRFFuser,
    WeightedFuser,
)
from knowledge_index.retrieval.pipeline import HybridPipeline, build_default_pipeline
from knowledge_index.retrieval.post import (
    DEFAULT_MMR_LAMBDA,
    DeduplicatorPostProcessor,
    MMRDiversifier,
    ParentExpander,
    PostProcessor,
    SpanExtractor,
)
from knowledge_index.retrieval.query_ops import (
    Decomposer,
    HyDEExpander,
    QueryOp,
    Rewriter,
    Stepback,
    apply_query_ops,
)
from knowledge_index.retrieval.reranking import (
    CohereReranker,
    LLMReranker,
    NullReranker,
    Reranker,
    VoyageReranker,
)
from knowledge_index.retrieval.retrievers import (
    DenseRetriever,
    Retriever,
    SparseBM25Retriever,
    SupportsEmbedQuery,
    SupportsSearch,
    gather_retrievers,
)

__all__ = [
    # pipeline
    "HybridPipeline",
    "build_default_pipeline",
    # query ops
    "QueryOp",
    "Rewriter",
    "HyDEExpander",
    "Decomposer",
    "Stepback",
    "apply_query_ops",
    # retrievers
    "Retriever",
    "DenseRetriever",
    "SparseBM25Retriever",
    "SupportsSearch",
    "SupportsEmbedQuery",
    "gather_retrievers",
    # fusion
    "Fuser",
    "RRFFuser",
    "WeightedFuser",
    "DEFAULT_RRF_K",
    # reranking
    "Reranker",
    "CohereReranker",
    "NullReranker",
    "VoyageReranker",
    "LLMReranker",
    # post-processing
    "PostProcessor",
    "MMRDiversifier",
    "ParentExpander",
    "DeduplicatorPostProcessor",
    "SpanExtractor",
    "DEFAULT_MMR_LAMBDA",
]
