"""Chunker contract (SPEC §7.2).

A chunker turns a :class:`ParsedDoc` into retrievable :class:`Chunk` objects.
Every chunker exposes ``name`` and a deterministic ``config`` dict so the
self-improvement loop (SPEC §8) can sweep its parameters.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable

from common.schemas import Chunk
from knowledge_index.ingestion.base import ParsedDoc


@runtime_checkable
class Chunker(Protocol):
    """Split a parsed document into chunks."""

    name: str
    config: dict[str, Any]

    def chunk(self, doc: ParsedDoc) -> list[Chunk]: ...


def make_chunk_id(doc_id: str, index: int, text: str) -> str:
    """Deterministic chunk id from doc id, ordinal, and content hash.

    Content-addressed so re-ingesting an unchanged doc yields identical ids
    (idempotent upserts in the index).
    """
    h = hashlib.sha256(f"{doc_id}:{index}:{text}".encode()).hexdigest()[:16]
    return f"{doc_id}::chunk-{index:04d}-{h}"


def build_chunks(
    doc: ParsedDoc,
    texts: list[str],
    *,
    extra_metadata: list[dict[str, Any]] | None = None,
) -> list[Chunk]:
    """Assemble :class:`Chunk` objects from split text, inheriting doc metadata.

    Each chunk gets ``parent_id = doc.doc_id`` (parent-child retrieval), the
    doc's ACL (copied so later mutation is isolated), and a char ``span`` within
    the parent document text when locatable.
    """
    acl = list(doc.metadata.get("acl", []))
    chunks: list[Chunk] = []
    cursor = 0
    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
        start = doc.text.find(text, cursor)
        span = (start, start + len(text)) if start >= 0 else None
        if start >= 0:
            cursor = start + len(text)
        meta: dict[str, Any] = {
            "doc_title": doc.metadata.get("title"),
            "source_path": doc.metadata.get("path"),
        }
        if extra_metadata and i < len(extra_metadata):
            meta.update(extra_metadata[i])
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc.doc_id, i, text),
                doc_id=doc.doc_id,
                parent_id=doc.doc_id,
                text=text,
                metadata={"span": span, **meta},
                acl=acl,
            )
        )
    return chunks


__all__ = ["Chunker", "make_chunk_id", "build_chunks"]
