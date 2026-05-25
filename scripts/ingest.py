"""scripts/ingest.py — knowledge-index ingestion CLI (SPEC §7, §14).

Usage::

    uv run scripts/ingest.py --src ./docs --collection main
    uv run scripts/ingest.py --src ./docs --collection main --embedder hash

Reads pipeline defaults from the Hydra ``default`` config (``index:`` section)
and infra/secrets from :class:`common.settings.Settings`. The ``hash`` embedder
needs no API keys and is the offline default when none are configured.
"""

from __future__ import annotations

import argparse
import asyncio

from common.config import load_config, to_container
from common.settings import get_settings
from harness.cache.embedding_cache import EmbeddingCache
from knowledge_index.indexing import QdrantIndex
from knowledge_index.pipeline import build_pipeline_from_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest a directory tree into the knowledge index.")
    p.add_argument("--src", required=True, help="Source file or directory to ingest.")
    p.add_argument("--collection", default="main", help="Qdrant collection name.")
    p.add_argument("--config", default="default", help="Hydra config name.")
    p.add_argument("--embedder", default=None, help="Override embedder (e.g. 'hash').")
    p.add_argument(
        "--enricher",
        default=None,
        help="Override enricher (e.g. 'null'/'title' for offline runs without an LLM key).",
    )
    p.add_argument(
        "--acl",
        default=None,
        help="Comma-separated principals to attach as the ACL for ingested chunks.",
    )
    p.add_argument(
        "--in-memory",
        action="store_true",
        help="Use an in-memory Qdrant (no server); useful for smoke tests.",
    )
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    cfg = to_container(load_config(args.config))
    idx_cfg = dict(cfg.get("index", {}))

    if args.embedder:
        # Fresh dict so a stale `dim` from the default config can't mismatch the
        # overriding embedder (e.g. hash defaults to 256, not voyage's 1024).
        idx_cfg["embedder"] = {"name": args.embedder}
    if args.enricher:
        idx_cfg["enricher"] = {"name": args.enricher}

    emb_cfg = idx_cfg.get("embedder", {})
    embedder_name = emb_cfg.get("name", "hash")
    default_dim = 256 if embedder_name == "hash" else 1024
    dim = int(emb_cfg.get("dim", default_dim))

    index_kwargs: dict = {"collection": args.collection, "dim": dim}
    if args.in_memory:
        index_kwargs["location"] = ":memory:"
    else:
        from qdrant_client import AsyncQdrantClient

        index_kwargs["client"] = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
    index = QdrantIndex(**index_kwargs)

    cache = EmbeddingCache(settings.embedding_cache_path)
    pipeline = build_pipeline_from_config({"index": idx_cfg}, index=index, embedding_cache=cache)

    acl = args.acl.split(",") if args.acl else None
    stats = await pipeline.ingest_dir(args.src, acl=acl)
    print(
        f"Ingested {stats.docs} docs → {stats.chunks} chunks into "
        f"'{args.collection}' (embedder={embedder_name}, dim={dim}). "
        f"Near-dup clusters: {stats.near_dup_clusters} ({stats.near_dup_docs} docs)."
    )
    cache.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
