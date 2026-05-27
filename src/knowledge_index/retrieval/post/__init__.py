"""Post-processing: MMR, parent expansion, dedup, reorder, spans (SPEC §7.6.6)."""

from knowledge_index.retrieval.post.base import PostProcessor, cosine
from knowledge_index.retrieval.post.dedup import DeduplicatorPostProcessor
from knowledge_index.retrieval.post.mmr import (
    DEFAULT_MMR_LAMBDA,
    EmbedFn,
    MMRDiversifier,
)
from knowledge_index.retrieval.post.parent import FetchParentFn, ParentExpander
from knowledge_index.retrieval.post.reorder import LostInTheMiddleReorder
from knowledge_index.retrieval.post.span import DEFAULT_MAX_SENTENCES, SpanExtractor

__all__ = [
    "PostProcessor",
    "cosine",
    "DeduplicatorPostProcessor",
    "DEFAULT_MMR_LAMBDA",
    "EmbedFn",
    "MMRDiversifier",
    "FetchParentFn",
    "ParentExpander",
    "LostInTheMiddleReorder",
    "DEFAULT_MAX_SENTENCES",
    "SpanExtractor",
]
