"""End-to-end ingestion pipeline test (SPEC §7.1–§7.5).

Fully offline: markdown parser, recursive chunker, null/title enricher, hash
embedder, in-memory Qdrant. Exercises the acceptance path of ka-tty without any
external services.
"""

from knowledge_index.chunking import RecursiveChunker
from knowledge_index.embedding import HashEmbedder
from knowledge_index.enrichment import TitleEnricher
from knowledge_index.indexing import QdrantIndex
from knowledge_index.pipeline import IngestionPipeline, build_pipeline_from_config


def _write_corpus(tmp_path, n: int = 5):
    for i in range(n):
        body = "\n\n".join(f"Paragraph {j} of document {i} discussing topic {i}." for j in range(6))
        (tmp_path / f"doc{i}.md").write_text(f"# Document {i}\n\n{body}\n")
    # a near-duplicate of doc0
    (tmp_path / "dup.md").write_text((tmp_path / "doc0.md").read_text() + "\n\nExtra line.")


async def test_pipeline_ingests_directory(tmp_path):
    _write_corpus(tmp_path, n=5)
    dim = 64
    index = QdrantIndex("e2e", dim=dim, location=":memory:")
    pipeline = IngestionPipeline(
        chunker=RecursiveChunker(chunk_size=200, chunk_overlap=20),
        enricher=TitleEnricher(),
        embedder=HashEmbedder(dim=dim),
        index=index,
    )
    stats = await pipeline.ingest_dir(tmp_path, acl=["team"])
    assert stats.docs == 6
    assert stats.chunks > 0
    # the near-duplicate is flagged (doc0 vs dup)
    assert stats.near_dup_clusters >= 1
    assert await index.count() == stats.chunks

    # retrieval works end to end and respects ACL
    emb = HashEmbedder(dim=dim)
    qvec = await emb.embed_query("topic 3 document")
    hits = await index.search_dense(qvec, k=5, filters={"user_principals": ["team"]})
    assert hits
    # outsider sees nothing (all chunks have non-empty acl)
    none = await index.search_dense(qvec, k=5, filters={"user_principals": ["stranger"]})
    assert none == []


async def test_build_pipeline_from_config(tmp_path):
    _write_corpus(tmp_path, n=2)
    index = QdrantIndex("cfg", dim=256, location=":memory:")
    cfg = {
        "index": {
            "chunker": {"name": "recursive", "chunk_size": 300, "chunk_overlap": 30},
            "enricher": {"name": "null"},
            "embedder": {"name": "hash"},
        }
    }
    pipeline = build_pipeline_from_config(cfg, index=index)
    stats = await pipeline.ingest_dir(tmp_path)
    assert stats.docs == 2
    assert await index.count() == stats.chunks
