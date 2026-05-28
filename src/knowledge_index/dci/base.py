"""DCI schemas + :class:`CorpusStore` protocol + in-memory backing (SPEC §15.1).

The DCI tools (grep / glob / ls / read / describe / neighbors) all return
typed, citation-bearing results. The schemas in this module are those return
types. They are deliberately separate from :class:`common.schemas.Chunk` and
:class:`common.schemas.Source` because DCI works at the *doc* level (a logical
unit a human would navigate to), while retrieval works at the *chunk* level.
Each schema still carries a :class:`Source` so the citation enforcer (SPEC
§6.6) can pin generated claims to the underlying chunk.

:class:`CorpusStore` is the read surface the tools call into; it is the line
across which prod swaps the dev :class:`InMemoryCorpusStore` (chunk-backed)
for a real corpus store (e.g. on-disk source store + Qdrant scrolls). All
operations take an explicit ``principals`` set and apply ACL filtering
*inside* the store, so a tool can never accidentally return a doc whose ACL
the caller doesn't intersect (SPEC §11 #6).
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from common.schemas import Chunk, Source

# ---------------------------------------------------------------------------
# Return-value schemas
# ---------------------------------------------------------------------------


class DocRef(BaseModel):
    """A pointer to a doc in the corpus — what ``glob`` and ``ls`` return."""

    doc_id: str
    path: str  # logical tree path, e.g. ``/collection/source/doc_id``
    title: str | None = None
    source: str | None = None  # source label (file path, URL, collection name)
    type: str | None = None  # short type hint: ``md``, ``pdf``, ``code``, …
    length: int | None = None  # total char count if known
    acl: list[str] = Field(default_factory=list)


class DocSlice(BaseModel):
    """A windowed read of a doc — what ``read`` returns."""

    doc_id: str
    content: str
    start_line: int = 1
    end_line: int | None = None
    truncated: bool = False  # True when ``max_bytes`` clipped the slice
    citation: Source  # for the citation enforcer (SPEC §6.6)


class DocMetadata(BaseModel):
    """Per-doc metadata — what ``describe`` returns."""

    doc_id: str
    title: str | None = None
    source: str | None = None
    authors: list[str] = Field(default_factory=list)
    length: int = 0
    acl: list[str] = Field(default_factory=list)
    ingested_at: str | None = None  # ISO8601 timestamp if known
    metadata: dict[str, Any] = Field(default_factory=dict)


class GrepHit(BaseModel):
    """One match from :class:`CorpusGrepTool` — line + context + citation."""

    doc_id: str
    line_no: int
    snippet: str  # the matching line
    context_before: list[str] = Field(default_factory=list)
    context_after: list[str] = Field(default_factory=list)
    citation: Source


class ChunkRef(BaseModel):
    """A KG-neighbor chunk — what ``neighbors`` returns."""

    chunk_id: str
    doc_id: str
    text: str
    hops: int = Field(ge=0)
    citation: Source


class DirectoryEntry(BaseModel):
    """One row of a :class:`DirectoryListing`."""

    name: str
    path: str
    kind: str  # "dir" or "doc"
    doc_id: str | None = None  # only set for kind="doc"


class DirectoryListing(BaseModel):
    """A directory-style view of the logical tree — what ``ls`` returns."""

    path: str
    entries: list[DirectoryEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CorpusStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CorpusStore(Protocol):
    """Read-only corpus surface the DCI tools call into.

    Implementations MUST apply ACL filtering inside each method against the
    caller's ``principals`` set. The contract: a doc is visible iff its ACL is
    empty (public) OR a principal intersects it. ``principals=None`` is treated
    as "no principals" — only public docs are visible.
    """

    async def glob(
        self,
        pattern: str = "**/*",
        *,
        types: list[str] | None = None,
        limit: int = 200,
        principals: list[str] | None = None,
    ) -> list[DocRef]: ...

    async def ls(
        self, path: str = "/", *, principals: list[str] | None = None
    ) -> DirectoryListing: ...

    async def read(
        self,
        doc_id: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        max_bytes: int = 50_000,
        principals: list[str] | None = None,
    ) -> DocSlice: ...

    async def describe(
        self, doc_id: str, *, principals: list[str] | None = None
    ) -> DocMetadata: ...

    async def grep(
        self,
        pattern: str,
        *,
        glob: str = "**/*",
        regex: bool = True,
        max_hits: int = 50,
        context_lines: int = 2,
        principals: list[str] | None = None,
    ) -> list[GrepHit]: ...


# ---------------------------------------------------------------------------
# In-memory implementation backed by indexed chunks
# ---------------------------------------------------------------------------


class _DocRecord(BaseModel):
    """Internal: a reassembled doc the in-memory store serves from chunks."""

    doc_id: str
    path: str
    text: str
    title: str | None = None
    source: str | None = None
    type: str | None = None
    authors: list[str] = Field(default_factory=list)
    ingested_at: str | None = None
    acl: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Map chunk_id -> (start_char, end_char) within ``text`` so we can pin a
    # GrepHit / DocSlice line back to the chunk that contains it.
    chunks: list[tuple[Chunk, int, int]] = Field(default_factory=list)


def _acl_visible(doc_acl: list[str], principals: list[str] | None) -> bool:
    """ACL gate: public iff empty acl, else any principal must intersect."""
    if not doc_acl:
        return True
    if not principals:
        return False
    return bool(set(doc_acl) & set(principals))


def _path_for(collection: str | None, source: str | None, doc_id: str) -> str:
    """Build the logical tree path ``/collection/source/doc_id`` (skip blanks)."""
    parts = [p for p in (collection, source, doc_id) if p]
    return "/" + "/".join(parts)


class InMemoryCorpusStore:
    """Dev :class:`CorpusStore` that reassembles docs from indexed chunks.

    The store ingests :class:`Chunk` objects: chunks sharing a ``doc_id`` are
    concatenated in order to form the doc text, and any chunk-level metadata
    that describes the *doc* (``title``, ``source``, ``collection``, ``type``,
    ``authors``, ``ingested_at``, ``path``) is lifted to the doc record. ACL
    on the doc is the union of its chunks' ACLs — a principal must be on at
    least one chunk's ACL to see the doc.
    """

    def __init__(self) -> None:
        self._docs: dict[str, _DocRecord] = {}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index a batch of chunks. Chunks are assumed pre-ordered per doc."""
        # Group by doc and preserve insertion order so the doc text matches the
        # order chunks arrived in. The chunking pipeline (SPEC §7.2) emits
        # chunks in document order, so this preserves source order.
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)

        for doc_id, doc_chunks in by_doc.items():
            existing = self._docs.get(doc_id)
            # Use the first chunk's metadata as the doc-level view.
            head = doc_chunks[0].metadata or {}
            collection = head.get("collection")
            source = head.get("source") or head.get("source_path")
            title = head.get("title")
            doc_type = head.get("type") or head.get("mime")
            authors = list(head.get("authors") or [])
            ingested_at = head.get("ingested_at")
            extra = {
                k: v
                for k, v in head.items()
                if k
                not in {
                    "collection",
                    "source",
                    "source_path",
                    "title",
                    "type",
                    "mime",
                    "authors",
                    "ingested_at",
                }
            }

            # Build text + chunk offsets. If the doc already existed, append.
            if existing is None:
                text = ""
                offsets: list[tuple[Chunk, int, int]] = []
                acl: set[str] = set()
            else:
                text = existing.text
                offsets = list(existing.chunks)
                acl = set(existing.acl)

            for chunk in doc_chunks:
                start = len(text)
                # Separate concatenated chunks with a single newline so line
                # numbering stays meaningful (each chunk starts on a fresh line
                # unless it already ends with one).
                if text and not text.endswith("\n"):
                    text += "\n"
                    start += 1
                text += chunk.text
                end = len(text)
                offsets.append((chunk, start, end))
                acl.update(chunk.acl)

            path = _path_for(collection, source, doc_id)
            self._docs[doc_id] = _DocRecord(
                doc_id=doc_id,
                path=path,
                text=text,
                title=title,
                source=source,
                type=doc_type,
                authors=authors,
                ingested_at=ingested_at,
                acl=sorted(acl),
                metadata=extra,
                chunks=offsets,
            )

    # --- helpers ---

    def _visible(self, principals: list[str] | None) -> list[_DocRecord]:
        return [d for d in self._docs.values() if _acl_visible(d.acl, principals)]

    def _chunk_for_offset(self, doc: _DocRecord, char_offset: int) -> Chunk | None:
        """Find the chunk whose ``[start, end)`` covers ``char_offset``."""
        for chunk, start, end in doc.chunks:
            if start <= char_offset < end:
                return chunk
        # Fall back to last chunk if past end (e.g. trailing newline land).
        return doc.chunks[-1][0] if doc.chunks else None

    def _line_to_char(self, text: str, line_no: int) -> int:
        """Char offset of the start of 1-indexed ``line_no``. Clamps to bounds."""
        if line_no <= 1:
            return 0
        # Walk forward counting newlines. Cheap because docs are bounded
        # (chunked corpora; not gigabyte files).
        pos = 0
        seen = 1
        while seen < line_no and pos < len(text):
            nl = text.find("\n", pos)
            if nl == -1:
                return len(text)
            pos = nl + 1
            seen += 1
        return pos

    # --- protocol implementation ---

    async def glob(
        self,
        pattern: str = "**/*",
        *,
        types: list[str] | None = None,
        limit: int = 200,
        principals: list[str] | None = None,
    ) -> list[DocRef]:
        type_set = {t.lower() for t in types} if types else None
        # fnmatch handles ``*`` and ``?``; ``**`` is treated as ``*`` here since
        # paths are shallow (collection / source / doc). Callers wanting strict
        # recursive globbing should swap in a CorpusStore backed by a real FS.
        normalized = pattern.replace("**/", "*").replace("/**", "/*")
        out: list[DocRef] = []
        for doc in self._visible(principals):
            if type_set is not None and (doc.type or "").lower() not in type_set:
                continue
            if not (fnmatch.fnmatch(doc.path, normalized) or fnmatch.fnmatch(doc.doc_id, pattern)):
                continue
            out.append(
                DocRef(
                    doc_id=doc.doc_id,
                    path=doc.path,
                    title=doc.title,
                    source=doc.source,
                    type=doc.type,
                    length=len(doc.text),
                    acl=list(doc.acl),
                )
            )
            if len(out) >= max(0, limit):
                break
        return out

    async def ls(self, path: str = "/", *, principals: list[str] | None = None) -> DirectoryListing:
        prefix = path if path.endswith("/") else path + "/"
        if path == "/":
            prefix = "/"

        entries: dict[str, DirectoryEntry] = {}
        for doc in self._visible(principals):
            if not doc.path.startswith(prefix):
                continue
            rest = doc.path[len(prefix) :]
            if not rest:
                continue
            head, _, tail = rest.partition("/")
            entry_path = prefix + head
            if tail:
                # Sub-directory under ``path``.
                entries.setdefault(head, DirectoryEntry(name=head, path=entry_path, kind="dir"))
            else:
                entries[head] = DirectoryEntry(
                    name=head, path=entry_path, kind="doc", doc_id=doc.doc_id
                )

        # Deterministic order: directories first, then docs, both alpha-sorted.
        ordered = sorted(entries.values(), key=lambda e: (e.kind != "dir", e.name))
        return DirectoryListing(path=path, entries=ordered)

    async def read(
        self,
        doc_id: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        max_bytes: int = 50_000,
        principals: list[str] | None = None,
    ) -> DocSlice:
        doc = self._docs.get(doc_id)
        if doc is None or not _acl_visible(doc.acl, principals):
            # Caller can't see this doc — return an empty slice rather than
            # leak existence. The empty content + a citation pointing nowhere
            # is the negative-confirmation contract.
            return DocSlice(
                doc_id=doc_id,
                content="",
                start_line=start_line,
                end_line=end_line,
                truncated=False,
                citation=Source(doc_id=doc_id, chunk_id="", metadata={"hidden": True}),
            )

        start_char = self._line_to_char(doc.text, max(1, start_line))
        if end_line is not None:
            # ``end_line`` is inclusive; advance to the start of the line after.
            end_char = self._line_to_char(doc.text, end_line + 1)
        else:
            end_char = len(doc.text)

        truncated = False
        if end_char - start_char > max_bytes:
            end_char = start_char + max_bytes
            truncated = True

        content = doc.text[start_char:end_char]
        anchor_chunk = self._chunk_for_offset(doc, start_char) or doc.chunks[0][0]
        citation = Source(
            doc_id=doc.doc_id,
            chunk_id=anchor_chunk.chunk_id,
            parent_id=anchor_chunk.parent_id,
            title=doc.title,
            span=(start_char, end_char),
            metadata={"source": doc.source} if doc.source else {},
        )
        return DocSlice(
            doc_id=doc.doc_id,
            content=content,
            start_line=max(1, start_line),
            end_line=end_line,
            truncated=truncated,
            citation=citation,
        )

    async def describe(self, doc_id: str, *, principals: list[str] | None = None) -> DocMetadata:
        doc = self._docs.get(doc_id)
        if doc is None or not _acl_visible(doc.acl, principals):
            # Same no-leak posture as ``read``.
            return DocMetadata(doc_id=doc_id, metadata={"hidden": True})
        return DocMetadata(
            doc_id=doc.doc_id,
            title=doc.title,
            source=doc.source,
            authors=list(doc.authors),
            length=len(doc.text),
            acl=list(doc.acl),
            ingested_at=doc.ingested_at,
            metadata=dict(doc.metadata),
        )

    async def grep(
        self,
        pattern: str,
        *,
        glob: str = "**/*",
        regex: bool = True,
        max_hits: int = 50,
        context_lines: int = 2,
        principals: list[str] | None = None,
    ) -> list[GrepHit]:
        if not pattern:
            return []
        try:
            matcher = re.compile(pattern, re.MULTILINE) if regex else re.compile(re.escape(pattern))
        except re.error:
            # Fall back to literal match on a malformed regex — never raise out
            # of a tool boundary; the sandbox executor catches but the policy
            # is "best-effort, never crash the graph" (SPEC §6.7).
            matcher = re.compile(re.escape(pattern))

        candidates = await self.glob(glob, limit=10_000, principals=principals)
        candidate_ids = {c.doc_id for c in candidates}
        ctx = max(0, context_lines)
        hits: list[GrepHit] = []
        for doc in self._visible(principals):
            if doc.doc_id not in candidate_ids:
                continue
            lines = doc.text.splitlines()
            for i, line in enumerate(lines, start=1):
                if not matcher.search(line):
                    continue
                # Pin the citation to whichever chunk this line lives in so the
                # downstream evidence package can attribute it (SPEC §15.5 #5).
                char_off = self._line_to_char(doc.text, i)
                anchor = self._chunk_for_offset(doc, char_off) or doc.chunks[0][0]
                hits.append(
                    GrepHit(
                        doc_id=doc.doc_id,
                        line_no=i,
                        snippet=line,
                        context_before=lines[max(0, i - 1 - ctx) : i - 1],
                        context_after=lines[i : i + ctx],
                        citation=Source(
                            doc_id=doc.doc_id,
                            chunk_id=anchor.chunk_id,
                            parent_id=anchor.parent_id,
                            title=doc.title,
                            span=(char_off, char_off + len(line)),
                            metadata={"line_no": i},
                        ),
                    )
                )
                if len(hits) >= max_hits:
                    return hits
        return hits


__all__ = [
    "DocRef",
    "DocSlice",
    "DocMetadata",
    "GrepHit",
    "ChunkRef",
    "DirectoryEntry",
    "DirectoryListing",
    "CorpusStore",
    "InMemoryCorpusStore",
]
