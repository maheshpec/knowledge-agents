"""Ingestion contracts: the ``Parser`` protocol and ``ParsedDoc`` (SPEC §7.1).

A parser turns raw bytes of a known MIME type into a normalized ``ParsedDoc``:
clean text plus a structural outline (headings, tables, code, figures) and
extracted metadata. Concrete parsers live in :mod:`knowledge_index.ingestion.parsers`.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from common.types import MimeType

StructureKind = Literal["heading", "paragraph", "table", "code", "figure", "list"]


class StructureElement(BaseModel):
    """One structural unit of a parsed document, in document order."""

    kind: StructureKind
    text: str
    level: int | None = None  # heading depth (1-6); None for non-headings
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDoc(BaseModel):
    """The normalized output of a parser (SPEC §7.1).

    ``text`` is the full normalized body (markdown where structure matters);
    ``structure`` is the ordered outline used by structure-aware chunkers.
    """

    doc_id: str
    text: str
    structure: list[StructureElement] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Parser(Protocol):
    """Parse raw bytes of a given MIME type into a :class:`ParsedDoc`."""

    async def parse(self, blob: bytes, hint: MimeType) -> ParsedDoc: ...


__all__ = ["StructureKind", "StructureElement", "ParsedDoc", "Parser"]
