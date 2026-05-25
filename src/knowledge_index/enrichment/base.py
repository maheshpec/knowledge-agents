"""Enricher contract (SPEC §7.3).

An enricher attaches retrieval-improving context to each chunk (stored on
``Chunk.context``). The contextual variant uses an LLM with the full document
in scope; cheaper variants use titles/headers or are no-ops (baseline).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from common.schemas import Chunk
from knowledge_index.ingestion.base import ParsedDoc


@runtime_checkable
class Enricher(Protocol):
    """Add ``context`` to chunks to improve retrieval (SPEC §7.3)."""

    name: str

    async def enrich(self, doc: ParsedDoc, chunks: list[Chunk]) -> list[Chunk]: ...


def embedding_text(chunk: Chunk) -> str:
    """The text actually embedded/indexed: context prepended to chunk text.

    Contextual retrieval (SPEC §7.3) improves recall by embedding the generated
    context together with the chunk body.
    """
    if chunk.context:
        return f"{chunk.context}\n\n{chunk.text}"
    return chunk.text


__all__ = ["Enricher", "embedding_text"]
