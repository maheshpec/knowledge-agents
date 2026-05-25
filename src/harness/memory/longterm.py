"""Long-term memory (SPEC §6.3) — durable facts in a dedicated vector index.

Stored in a **separate** Qdrant collection (default ``ka_memory_longterm``), not a
namespace within the corpus collection: memory is per-user, reindexed on a
different cadence, and has a different ACL surface (SPEC §6.3). Reuses Convoy B's
``QdrantIndex``, embedder, and embedding cache.

Each :class:`MemoryItem` is stored as a :class:`Chunk` whose payload carries the
full item (so reads reconstruct it faithfully). Writes are expected to come
through the extraction step (``extraction.py``) — do not store everything.
"""

from __future__ import annotations

from collections.abc import Callable

from common.schemas import Chunk
from common.types import MemoryItem
from harness.observability.tracing import traced
from knowledge_index.embedding.base import Embedder
from knowledge_index.indexing.qdrant_index import QdrantIndex

DEFAULT_MEMORY_COLLECTION = "ka_memory_longterm"


def _embed_text(item: MemoryItem) -> str:
    """Text used to embed/retrieve a memory item (key gives it a handle)."""
    return f"{item.key}: {item.value}"


class LongTermMemory:
    """Vector-backed durable memory for one owner principal (SPEC §6.3)."""

    scope = "long_term"

    def __init__(
        self,
        index: QdrantIndex,
        embedder: Embedder,
        *,
        owner: str | None = None,
    ) -> None:
        self._index = index
        self._embedder = embedder
        # When set, the owner is the chunk ACL and the read principal — memory is
        # per-user, so cross-user reads return nothing (public if owner is None).
        self.owner = owner

    def _to_chunk(self, item: MemoryItem, embedding: list[float]) -> Chunk:
        return Chunk(
            chunk_id=item.key,
            doc_id="memory",
            text=str(item.value),
            embedding=embedding,
            metadata={"memory_item": item.model_dump(mode="json")},
            acl=[self.owner] if self.owner else [],
        )

    @staticmethod
    def _to_item(chunk: Chunk, *, score: float | None = None) -> MemoryItem:
        item = MemoryItem.model_validate(chunk.metadata["memory_item"])
        if score is not None:
            item.score = score
        return item

    @traced(span_name="memory.longterm.write")
    async def write(self, item: MemoryItem) -> None:
        vec = (await self._embedder.embed_documents([_embed_text(item)]))[0]
        await self._index.upsert([self._to_chunk(item, vec)])

    @traced(span_name="memory.longterm.read")
    async def read(self, query: str, k: int = 5) -> list[MemoryItem]:
        vec = await self._embedder.embed_query(query)
        filters = {"acl": [self.owner]} if self.owner else {}
        candidates = await self._index.search_dense(vec, k, filters)
        return [self._to_item(c.chunk, score=c.score) for c in candidates]

    async def all(self) -> list[MemoryItem]:
        chunks = await self._index.iter_chunks()
        if self.owner:
            # per-user view: keep this owner's items + any public (empty-acl) ones
            chunks = [c for c in chunks if not c.acl or self.owner in c.acl]
        return [self._to_item(c) for c in chunks]

    @traced(span_name="memory.longterm.forget")
    async def forget(self, predicate: Callable[[MemoryItem], bool]) -> None:
        drop = [it.key for it in await self.all() if predicate(it)]
        if drop:
            await self._index.delete(drop)


__all__ = ["LongTermMemory", "DEFAULT_MEMORY_COLLECTION"]
