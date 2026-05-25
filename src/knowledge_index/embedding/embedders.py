"""Concrete embedders (SPEC §7.4).

- ``VoyageEmbedder`` — ``voyage-3-large`` (1024-dim), primary.
- ``OpenAIEmbedder`` — ``text-embedding-3-large`` (3072-dim), fallback.
- ``HashEmbedder`` — deterministic local embedder (no network) for tests and
  offline dev; a stand-in for the ``bge-large`` local fallback in SPEC §7.4.

Network SDKs are imported lazily; ``HashEmbedder`` keeps the whole pipeline
runnable and unit-testable with no API keys.
"""

from __future__ import annotations

import hashlib
import math

from harness.cache.embedding_cache import EmbeddingCache
from knowledge_index.embedding.base import BaseEmbedder

# Known embedding dimensions per model (SPEC §7.4).
_MODEL_DIMS = {
    "voyage-3-large": 1024,
    "voyage-3": 1024,
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}


class VoyageEmbedder(BaseEmbedder):
    """Voyage embeddings via the ``voyageai`` SDK (lazy import)."""

    def __init__(
        self,
        model: str = "voyage-3-large",
        *,
        api_key: str | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        super().__init__(cache)
        self.name = model
        self.dim = _MODEL_DIMS.get(model, 1024)
        self._api_key = api_key
        self._client = None

    def _get_client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            import voyageai  # type: ignore

            self._client = voyageai.AsyncClient(api_key=self._api_key)
        return self._client

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        resp = await client.embed(texts, model=self.name, input_type="document")
        return [list(v) for v in resp.embeddings]

    async def embed_query(self, text: str) -> list[float]:
        client = self._get_client()
        resp = await client.embed([text], model=self.name, input_type="query")
        return list(resp.embeddings[0])


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings via the ``openai`` SDK (lazy import), fallback path."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        *,
        api_key: str | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        super().__init__(cache)
        self.name = model
        self.dim = _MODEL_DIMS.get(model, 3072)
        self._api_key = api_key
        self._client = None

    def _get_client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            from openai import AsyncOpenAI  # type: ignore

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        resp = await client.embeddings.create(model=self.name, input=texts)
        return [list(d.embedding) for d in resp.data]


class HashEmbedder(BaseEmbedder):
    """Deterministic, dependency-free embedder for tests and offline dev.

    Maps token hashes into a fixed-dimension L2-normalized vector. Not
    semantically meaningful, but stable and fast — lets the full ingest →
    index → search path run with no API keys.
    """

    def __init__(self, dim: int = 256, *, cache: EmbeddingCache | None = None) -> None:
        super().__init__(cache)
        self.name = f"hash-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [x / norm for x in vec]

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


__all__ = ["VoyageEmbedder", "OpenAIEmbedder", "HashEmbedder"]
