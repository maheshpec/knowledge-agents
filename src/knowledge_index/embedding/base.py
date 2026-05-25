"""Embedder contract + cache-wrapping base (SPEC §7.4, §6.12 tier 2).

Every embedder is wrapped through the SQLite :class:`EmbeddingCache` so identical
text under the same model is embedded exactly once. Concrete embedders implement
``_embed_raw``; the base handles cache lookup/fill and ordering.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness.cache.embedding_cache import EmbeddingCache


@runtime_checkable
class Embedder(Protocol):
    """Produce dense vectors for documents and queries (SPEC §7.4)."""

    name: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class BaseEmbedder:
    """Shared cache-aware embedder base.

    Subclasses set ``name``/``dim`` and implement :meth:`_embed_raw`. Queries are
    not cached by default (they are usually unique); documents always are.
    """

    name: str = "base"
    dim: int = 0

    def __init__(self, cache: EmbeddingCache | None = None) -> None:
        self._cache = cache

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._cache is None:
            return await self._embed_raw(texts)
        # Resolve cache hits; embed only the misses, preserving order.
        cached = self._cache.get_many(self.name, texts)
        misses = [t for t in texts if t not in cached]
        if misses:
            # De-duplicate misses so repeated text in one batch costs one call.
            unique_misses = list(dict.fromkeys(misses))
            vectors = await self._embed_raw(unique_misses)
            for t, v in zip(unique_misses, vectors, strict=True):
                self._cache.put(self.name, t, v)
                cached[t] = v
        return [cached[t] for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed_raw([text]))[0]


__all__ = ["Embedder", "BaseEmbedder"]
