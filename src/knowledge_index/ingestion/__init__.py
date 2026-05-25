"""Ingestion package (SPEC §7.1): parse → normalize → dedup.

Public surface re-exported here so the registry and CLI can import concrete
classes from ``knowledge_index.ingestion``.
"""

from __future__ import annotations

from knowledge_index.ingestion.base import (
    ParsedDoc,
    Parser,
    StructureElement,
    StructureKind,
)
from knowledge_index.ingestion.dedup import MinHashDeduplicator, iter_unique, jaccard
from knowledge_index.ingestion.normalize import (
    decode_bytes,
    detect_encoding,
    normalize_for_dedup,
    normalize_text,
)
from knowledge_index.ingestion.parsers import (
    DocxParser,
    HTMLParser,
    MarkdownParser,
    PDFParser,
    PlainTextParser,
    get_parser,
    mime_from_path,
    parse_blob,
    parse_path,
)

__all__ = [
    "ParsedDoc",
    "Parser",
    "StructureElement",
    "StructureKind",
    "MinHashDeduplicator",
    "jaccard",
    "iter_unique",
    "decode_bytes",
    "detect_encoding",
    "normalize_text",
    "normalize_for_dedup",
    "MarkdownParser",
    "PlainTextParser",
    "HTMLParser",
    "PDFParser",
    "DocxParser",
    "get_parser",
    "mime_from_path",
    "parse_blob",
    "parse_path",
]
