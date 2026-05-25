"""End-to-end ingestion pipeline (SPEC §7.1–§7.5).

Wires the Phase 1B stages into one callable:

    parse → normalize → dedup-flag → chunk → enrich → embed → index.upsert

Each stage is a swappable component (registry-constructed). The pipeline owns
no policy beyond ordering; defaults match ``configs/default.yaml`` ``index:``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from common.schemas import Chunk
from knowledge_index.chunking.base import Chunker
from knowledge_index.embedding.base import Embedder
from knowledge_index.enrichment.base import Enricher, embedding_text
from knowledge_index.indexing.base import Index
from knowledge_index.ingestion.base import ParsedDoc
from knowledge_index.ingestion.dedup import MinHashDeduplicator
from knowledge_index.ingestion.parsers import mime_from_path, parse_path

# File extensions the directory walker will attempt to ingest.
_INGESTABLE = {".pdf", ".docx", ".html", ".htm", ".md", ".markdown", ".txt", ".rst"}


class IngestStats(BaseModel):
    """Summary of an ingestion run."""

    docs: int = 0
    chunks: int = 0
    near_dup_clusters: int = 0
    near_dup_docs: int = 0


class IngestionPipeline:
    """Compose the Phase 1B stages around a target :class:`Index`."""

    def __init__(
        self,
        *,
        chunker: Chunker,
        enricher: Enricher,
        embedder: Embedder,
        index: Index,
        dedup_threshold: float = 0.9,
    ) -> None:
        self.chunker = chunker
        self.enricher = enricher
        self.embedder = embedder
        self.index = index
        self.dedup_threshold = dedup_threshold

    # --- discovery --------------------------------------------------------

    @staticmethod
    def discover(src: str | Path) -> list[Path]:
        """Recursively list ingestable files under ``src`` (sorted, stable)."""
        root = Path(src)
        if root.is_file():
            return [root]
        return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _INGESTABLE)

    # --- stages -----------------------------------------------------------

    async def _embed_chunks(self, chunks: list[Chunk]) -> None:
        vectors = await self.embedder.embed_documents([embedding_text(c) for c in chunks])
        for chunk, vec in zip(chunks, vectors, strict=True):
            chunk.embedding = vec

    async def ingest_docs(
        self, docs: list[ParsedDoc], *, acl: list[str] | None = None
    ) -> IngestStats:
        """Run dedup → chunk → enrich → embed → upsert over parsed docs."""
        # 1. Near-duplicate flagging (does not remove; records cluster id).
        dedup = MinHashDeduplicator(threshold=self.dedup_threshold)
        for d in docs:
            dedup.add(d.doc_id, d.text)
        clusters = dedup.clusters()
        doc_to_cluster = {doc_id: root for root, members in clusters.items() for doc_id in members}
        near_dup_docs = sum(len(m) for m in clusters.values())

        all_chunks: list[Chunk] = []
        for d in docs:
            if acl is not None:
                d.metadata.setdefault("acl", acl)
            if d.doc_id in doc_to_cluster:
                d.metadata["near_dup_cluster"] = doc_to_cluster[d.doc_id]
            chunks = self.chunker.chunk(d)
            chunks = await self.enricher.enrich(d, chunks)
            if d.doc_id in doc_to_cluster:
                for c in chunks:
                    c.metadata["near_dup_cluster"] = doc_to_cluster[d.doc_id]
            all_chunks.extend(chunks)

        if all_chunks:
            await self._embed_chunks(all_chunks)
            await self.index.upsert(all_chunks)

        return IngestStats(
            docs=len(docs),
            chunks=len(all_chunks),
            near_dup_clusters=len(clusters),
            near_dup_docs=near_dup_docs,
        )

    async def ingest_paths(
        self, paths: Iterable[str | Path], *, acl: list[str] | None = None
    ) -> IngestStats:
        """Parse files at ``paths`` then ingest them."""
        docs: list[ParsedDoc] = []
        for p in paths:
            # mime_from_path keeps unknown/binary out of the plain-text path noisily.
            _ = mime_from_path(p)
            docs.append(await parse_path(p))
        return await self.ingest_docs(docs, acl=acl)

    async def ingest_dir(self, src: str | Path, *, acl: list[str] | None = None) -> IngestStats:
        """Discover and ingest every supported file under ``src``."""
        return await self.ingest_paths(self.discover(src), acl=acl)


def build_pipeline_from_config(
    cfg: dict[str, Any],
    *,
    index: Index,
    embedding_cache: Any = None,
) -> IngestionPipeline:
    """Construct a pipeline from the ``index:`` section of a Hydra config dict."""
    from knowledge_index.chunking import build_chunker
    from knowledge_index.embedding import build_embedder
    from knowledge_index.enrichment import build_enricher

    idx_cfg = cfg.get("index", cfg)
    ch = dict(idx_cfg.get("chunker", {"name": "recursive"}))
    en = dict(idx_cfg.get("enricher", {"name": "null"}))
    em = dict(idx_cfg.get("embedder", {"name": "hash"}))

    chunker = build_chunker(ch.pop("name"), **ch)
    enricher = build_enricher(en.pop("name"), **en)
    em_name = em.pop("name")
    if em_name != "hash":
        em.pop("dim", None)  # dim is derived from the model for real embedders
    embedder = build_embedder(em_name, cache=embedding_cache, **em)
    return IngestionPipeline(chunker=chunker, enricher=enricher, embedder=embedder, index=index)


__all__ = ["IngestStats", "IngestionPipeline", "build_pipeline_from_config"]
