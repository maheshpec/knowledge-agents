"""Chunking package (SPEC §7.2).

Exposes the chunker strategies plus a small registry factory so the
self-improvement loop and the CLI can construct a chunker by name + params.
"""

from __future__ import annotations

from typing import Any

from knowledge_index.chunking.base import (
    Chunker,
    build_chunks,
    make_chunk_id,
    split_sentences,
)
from knowledge_index.chunking.chunkers import (
    LateChunkingChunker,
    MarkdownHeaderChunker,
    PropositionalChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceWindowChunker,
)

# Registry name -> class (mirrors configs/components.yaml `chunkers`).
CHUNKER_REGISTRY: dict[str, type] = {
    "recursive": RecursiveChunker,
    "markdown_header": MarkdownHeaderChunker,
    "semantic": SemanticChunker,
    "sentence_window": SentenceWindowChunker,
    "late_chunking": LateChunkingChunker,
    "propositional": PropositionalChunker,
}


def build_chunker(name: str, **params: Any) -> Chunker:
    """Instantiate a chunker by registry name with parameter overrides."""
    if name not in CHUNKER_REGISTRY:
        raise KeyError(f"unknown chunker '{name}'; known: {sorted(CHUNKER_REGISTRY)}")
    return CHUNKER_REGISTRY[name](**params)  # type: ignore[return-value]


__all__ = [
    "Chunker",
    "build_chunks",
    "make_chunk_id",
    "split_sentences",
    "RecursiveChunker",
    "MarkdownHeaderChunker",
    "SemanticChunker",
    "SentenceWindowChunker",
    "LateChunkingChunker",
    "PropositionalChunker",
    "CHUNKER_REGISTRY",
    "build_chunker",
]
