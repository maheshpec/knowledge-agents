"""Post-processing: MMR diversify, parent expansion, dedup (SPEC §7.6.6)."""

from knowledge_index.retrieval.post.base import PostProcessor, cosine
from knowledge_index.retrieval.post.dedup import DeduplicatorPostProcessor
from knowledge_index.retrieval.post.mmr import (
    DEFAULT_MMR_LAMBDA,
    EmbedFn,
    MMRDiversifier,
)
from knowledge_index.retrieval.post.parent import FetchParentFn, ParentExpander

__all__ = [
    "PostProcessor",
    "cosine",
    "DeduplicatorPostProcessor",
    "DEFAULT_MMR_LAMBDA",
    "EmbedFn",
    "MMRDiversifier",
    "FetchParentFn",
    "ParentExpander",
]
