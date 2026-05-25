"""Knowledge index (SPEC §7): ingestion → chunking → enrichment → embedding → indexing.

Registry class paths in ``configs/components.yaml`` resolve against these
subpackages (e.g. ``knowledge_index.chunking.RecursiveChunker``), so the
concrete components are re-exported from their package ``__init__`` modules.
"""

from __future__ import annotations

from knowledge_index.pipeline import (
    IngestionPipeline,
    IngestStats,
    build_pipeline_from_config,
)

__all__ = ["IngestionPipeline", "IngestStats", "build_pipeline_from_config"]
