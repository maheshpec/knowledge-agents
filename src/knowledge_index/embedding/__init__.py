"""Embedding package (SPEC §7.4): cache-wrapped dense embedders + registry."""

from __future__ import annotations

from typing import Any

from harness.cache.embedding_cache import EmbeddingCache
from knowledge_index.embedding.base import BaseEmbedder, Embedder
from knowledge_index.embedding.embedders import (
    HashEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
)

EMBEDDER_REGISTRY: dict[str, type] = {
    "voyage-3-large": VoyageEmbedder,
    "text-embedding-3-large": OpenAIEmbedder,
    "hash": HashEmbedder,
}


def build_embedder(name: str, *, cache: EmbeddingCache | None = None, **params: Any) -> Embedder:
    """Construct an embedder by registry name.

    For the model-named registry entries the model id is passed through so a
    single class serves a family (e.g. voyage-3 vs voyage-3-large).
    """
    if name in ("voyage-3-large", "text-embedding-3-large"):
        return EMBEDDER_REGISTRY[name](model=name, cache=cache, **params)  # type: ignore[return-value]
    if name == "hash":
        return HashEmbedder(cache=cache, **params)
    if name in EMBEDDER_REGISTRY:
        return EMBEDDER_REGISTRY[name](cache=cache, **params)  # type: ignore[return-value]
    raise KeyError(f"unknown embedder '{name}'; known: {sorted(EMBEDDER_REGISTRY)}")


__all__ = [
    "Embedder",
    "BaseEmbedder",
    "VoyageEmbedder",
    "OpenAIEmbedder",
    "HashEmbedder",
    "EMBEDDER_REGISTRY",
    "build_embedder",
]
