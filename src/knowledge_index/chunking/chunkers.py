"""Concrete chunking strategies (SPEC §7.2).

``RecursiveChunker`` and ``MarkdownHeaderChunker`` are the Phase 1 defaults and
use ``langchain_text_splitters`` (a core dependency). ``SemanticChunker`` is a
registry entry whose heavy dependency (``langchain_experimental``) is imported
lazily so the module imports cleanly without it.
"""

from __future__ import annotations

from typing import Any

from common.schemas import Chunk
from knowledge_index.chunking.base import build_chunks
from knowledge_index.ingestion.base import ParsedDoc

# Default markdown header levels to split on, coarse → fine.
_DEFAULT_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class RecursiveChunker:
    """Character-recursive splitter (SPEC §7.2 default: 500/75 on markdown)."""

    name = "recursive"

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.config: dict[str, Any] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": separators,
        }

    def _splitter(self):  # type: ignore[no-untyped-def]
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        kwargs: dict[str, Any] = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        if self.separators is not None:
            kwargs["separators"] = self.separators
        return RecursiveCharacterTextSplitter(**kwargs)

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        texts = self._splitter().split_text(doc.text)
        return build_chunks(doc, texts)


class MarkdownHeaderChunker:
    """Split by markdown headers, then recursively size each section (SPEC §7.2).

    Header path is recorded in each chunk's metadata so downstream packers and
    the ``TitleEnricher`` can reconstruct section context.
    """

    name = "markdown_header"

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
    ) -> None:
        self.headers_to_split_on = headers_to_split_on or _DEFAULT_HEADERS
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config: dict[str, Any] = {
            "headers_to_split_on": [h[0] for h in self.headers_to_split_on],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )
        sections = header_splitter.split_text(doc.text)
        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts: list[str] = []
        extra: list[dict[str, Any]] = []
        for section in sections:
            header_path = [v for v in section.metadata.values()]
            for piece in size_splitter.split_text(section.page_content):
                texts.append(piece)
                extra.append({"header_path": header_path})
        if not texts:
            # No headers present: fall back to plain recursive sizing.
            texts = size_splitter.split_text(doc.text)
            extra = [{} for _ in texts]
        return build_chunks(doc, texts, extra_metadata=extra)


class SemanticChunker:
    """Embedding-based semantic splitter (lazy ``langchain_experimental``)."""

    name = "semantic"

    def __init__(self, threshold: float = 0.75, embedder: Any = None) -> None:
        self.threshold = threshold
        self._embedder = embedder
        self.config: dict[str, Any] = {"threshold": threshold}

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        try:
            from langchain_experimental.text_splitter import SemanticChunker as _LCSemantic
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "SemanticChunker requires 'langchain_experimental'; install ingest extras"
            ) from e
        if self._embedder is None:
            raise ValueError("SemanticChunker requires an embeddings backend")
        splitter = _LCSemantic(
            self._embedder,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.threshold * 100,
        )
        texts = splitter.split_text(doc.text)
        return build_chunks(doc, texts)


__all__ = ["RecursiveChunker", "MarkdownHeaderChunker", "SemanticChunker"]
