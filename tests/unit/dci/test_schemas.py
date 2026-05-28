"""Tests for the DCI return-value schemas (SPEC §15.1)."""

from __future__ import annotations

from common.schemas import Source
from knowledge_index.dci import (
    ChunkRef,
    DirectoryEntry,
    DirectoryListing,
    DocMetadata,
    DocRef,
    DocSlice,
    GrepHit,
)


def test_doc_ref_minimal_construction():
    ref = DocRef(doc_id="d", path="/x/d")
    assert ref.title is None
    assert ref.acl == []


def test_doc_slice_carries_citation():
    src = Source(doc_id="d", chunk_id="d:0", span=(0, 10))
    slc = DocSlice(doc_id="d", content="hello", citation=src)
    assert slc.citation.span == (0, 10)
    assert slc.truncated is False


def test_grep_hit_carries_context_and_citation():
    src = Source(doc_id="d", chunk_id="d:0")
    hit = GrepHit(
        doc_id="d",
        line_no=3,
        snippet="match",
        context_before=["a", "b"],
        context_after=["c"],
        citation=src,
    )
    assert hit.line_no == 3
    assert hit.citation.doc_id == "d"


def test_chunk_ref_hops_non_negative():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChunkRef(
            chunk_id="c",
            doc_id="d",
            text="t",
            hops=-1,
            citation=Source(doc_id="d", chunk_id="c"),
        )


def test_directory_listing_round_trip():
    listing = DirectoryListing(
        path="/main",
        entries=[
            DirectoryEntry(name="docs", path="/main/docs", kind="dir"),
            DirectoryEntry(name="doc-a", path="/main/docs/doc-a", kind="doc", doc_id="doc-a"),
        ],
    )
    assert listing.entries[1].doc_id == "doc-a"


def test_doc_metadata_defaults():
    md = DocMetadata(doc_id="d")
    assert md.length == 0
    assert md.authors == []
    assert md.acl == []
