"""Tests for chunking strategies (SPEC §7.2)."""

from knowledge_index.chunking import (
    CHUNKER_REGISTRY,
    MarkdownHeaderChunker,
    RecursiveChunker,
    build_chunker,
)
from knowledge_index.ingestion.base import ParsedDoc


def _doc(text: str) -> ParsedDoc:
    return ParsedDoc(doc_id="doc-1", text=text, metadata={"title": "T", "acl": ["alice"]})


def test_recursive_chunker_splits_and_sets_parent():
    text = " ".join(f"token{i}" for i in range(400))
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=20).chunk(_doc(text))
    assert len(chunks) > 1
    assert all(c.parent_id == "doc-1" for c in chunks)
    assert all(c.acl == ["alice"] for c in chunks)
    # deterministic, content-addressed ids
    again = RecursiveChunker(chunk_size=200, chunk_overlap=20).chunk(_doc(text))
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]


def test_recursive_chunker_exposes_config():
    ch = RecursiveChunker(chunk_size=500, chunk_overlap=75)
    assert ch.config == {"chunk_size": 500, "chunk_overlap": 75, "separators": None}
    assert ch.name == "recursive"


def test_markdown_header_chunker_records_header_path():
    md = "# Top\n\n" + "alpha " * 50 + "\n\n## Sub\n\n" + "beta " * 50
    chunks = MarkdownHeaderChunker(chunk_size=200, chunk_overlap=10).chunk(_doc(md))
    assert chunks
    assert any(c.metadata.get("header_path") for c in chunks)


def test_markdown_header_chunker_falls_back_without_headers():
    chunks = MarkdownHeaderChunker().chunk(_doc("plain text no headers here"))
    assert len(chunks) == 1


def test_build_chunker_registry():
    assert set(CHUNKER_REGISTRY) >= {"recursive", "markdown_header", "semantic"}
    ch = build_chunker("recursive", chunk_size=300)
    assert ch.config["chunk_size"] == 300
