"""Tests for the in-memory corpus store (SPEC §15.1).

Cover the surface the DCI tools call into: glob / ls / read / describe / grep.
Verify ACL enforcement happens *inside* the store (a hidden doc never leaks via
``read`` / ``describe`` / ``grep`` / ``ls`` / ``glob``).
"""

from __future__ import annotations

import pytest

from common.schemas import Chunk
from knowledge_index.dci import InMemoryCorpusStore


def _chunk(
    doc_id: str,
    chunk_id: str,
    text: str,
    *,
    collection: str = "main",
    source: str = "docs",
    title: str | None = None,
    type: str | None = "md",
    acl: list[str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        acl=acl or [],
        metadata={
            "collection": collection,
            "source": source,
            "title": title,
            "type": type,
        },
    )


@pytest.fixture
def store() -> InMemoryCorpusStore:
    s = InMemoryCorpusStore()
    s.add_chunks(
        [
            _chunk("doc-a", "doc-a:0", "Alpha line one\nAlpha line two\nAlpha line three"),
            _chunk("doc-a", "doc-a:1", "Alpha line four\nAlpha line five"),
            _chunk("doc-b", "doc-b:0", "Beta one\nBeta two", source="papers"),
            _chunk(
                "doc-secret",
                "doc-secret:0",
                "Top secret content with keyword foobar",
                source="restricted",
                acl=["team-x"],
            ),
        ]
    )
    return s


# --- ACL ----


async def test_glob_hides_private_docs_when_no_principals(store: InMemoryCorpusStore):
    refs = await store.glob("**/*")
    ids = {r.doc_id for r in refs}
    assert "doc-a" in ids and "doc-b" in ids
    assert "doc-secret" not in ids  # ACL-gated


async def test_glob_reveals_private_docs_to_intersecting_principal(store: InMemoryCorpusStore):
    refs = await store.glob("**/*", principals=["team-x"])
    ids = {r.doc_id for r in refs}
    assert "doc-secret" in ids


async def test_read_returns_empty_for_unauthorized_caller(store: InMemoryCorpusStore):
    slc = await store.read("doc-secret")
    assert slc.content == ""
    assert slc.citation.metadata.get("hidden") is True


async def test_describe_returns_empty_for_unauthorized_caller(store: InMemoryCorpusStore):
    md = await store.describe("doc-secret")
    assert md.title is None
    assert md.metadata.get("hidden") is True


async def test_grep_skips_unauthorized_docs(store: InMemoryCorpusStore):
    hits = await store.grep("foobar")
    assert hits == []
    hits = await store.grep("foobar", principals=["team-x"])
    assert len(hits) == 1
    assert hits[0].doc_id == "doc-secret"


# --- glob / types / limit ----


async def test_glob_type_filter_matches_lowercase(store: InMemoryCorpusStore):
    refs = await store.glob("**/*", types=["md"])
    assert {r.doc_id for r in refs} == {"doc-a", "doc-b"}
    refs = await store.glob("**/*", types=["pdf"])
    assert refs == []


async def test_glob_limit_clips_result(store: InMemoryCorpusStore):
    refs = await store.glob("**/*", limit=1)
    assert len(refs) == 1


# --- ls ----


async def test_ls_root_lists_collections(store: InMemoryCorpusStore):
    listing = await store.ls("/")
    names = [e.name for e in listing.entries]
    assert names == ["main"]  # only the single visible collection


async def test_ls_descent_to_doc(store: InMemoryCorpusStore):
    listing = await store.ls("/main/docs/")
    kinds = {e.kind for e in listing.entries}
    assert "doc" in kinds
    docs = {e.doc_id for e in listing.entries if e.kind == "doc"}
    assert docs == {"doc-a"}  # only the docs in this source


# --- read ----


async def test_read_returns_full_text_with_citation(store: InMemoryCorpusStore):
    slc = await store.read("doc-a")
    assert "Alpha line one" in slc.content
    assert "Alpha line five" in slc.content
    assert slc.citation.doc_id == "doc-a"
    assert slc.citation.chunk_id  # first chunk anchors the slice


async def test_read_windowed_by_lines(store: InMemoryCorpusStore):
    slc = await store.read("doc-a", start_line=2, end_line=3)
    assert slc.content.startswith("Alpha line two")
    assert "Alpha line three" in slc.content
    assert "Alpha line one" not in slc.content


async def test_read_truncates_by_max_bytes(store: InMemoryCorpusStore):
    slc = await store.read("doc-a", max_bytes=5)
    assert len(slc.content) == 5
    assert slc.truncated is True


# --- describe ----


async def test_describe_returns_aggregated_metadata(store: InMemoryCorpusStore):
    md = await store.describe("doc-a")
    assert md.length > 0
    assert md.acl == []  # public doc


# --- grep ----


async def test_grep_returns_hits_with_context(store: InMemoryCorpusStore):
    hits = await store.grep("line two", context_lines=1)
    assert len(hits) == 1
    h = hits[0]
    assert h.doc_id == "doc-a"
    assert h.line_no == 2
    assert h.snippet == "Alpha line two"
    assert h.context_before == ["Alpha line one"]
    assert h.context_after == ["Alpha line three"]


async def test_grep_pin_citation_to_owning_chunk(store: InMemoryCorpusStore):
    # ``Alpha line four`` lives in the second chunk; verify the citation pins to it.
    hits = await store.grep("line four")
    assert len(hits) == 1
    assert hits[0].citation.chunk_id == "doc-a:1"


async def test_grep_literal_mode_ignores_regex_metacharacters(store: InMemoryCorpusStore):
    # ``.`` should NOT match arbitrary chars when regex=False.
    hits = await store.grep("line.two", regex=False)
    assert hits == []


async def test_grep_max_hits_caps_results(store: InMemoryCorpusStore):
    hits = await store.grep("Alpha", max_hits=2)
    assert len(hits) == 2


async def test_grep_malformed_regex_falls_back_to_literal(store: InMemoryCorpusStore):
    # An unterminated group must not raise; the tool boundary stays clean.
    hits = await store.grep("(unterminated")
    assert hits == []  # literal "(unterminated" not present, no crash
