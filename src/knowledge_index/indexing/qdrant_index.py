"""Qdrant hybrid index (SPEC §7.5).

One collection holds a named ``dense`` vector and a named ``sparse`` vector per
chunk. ACLs are enforced *at search time* via payload filters (never after
retrieval). Snapshots are taken by scrolling points to a local JSON file so
index versioning works in both in-memory dev mode and against a server; a
production deployment can swap in Qdrant's native snapshot API.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from common.errors import KnowledgeAgentError
from common.schemas import Chunk, RetrievalCandidate
from knowledge_index.enrichment.base import embedding_text
from knowledge_index.indexing.base import (
    HashingBM25Encoder,
    SnapshotRef,
    SparseEncoder,
)

DENSE = "dense"
SPARSE = "sparse"


def _point_id(chunk_id: str) -> str:
    """Map an arbitrary string chunk id to a stable UUID Qdrant accepts."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantIndex:
    """Hybrid (dense + sparse) Qdrant-backed index with ACL filtering."""

    def __init__(
        self,
        collection: str,
        dim: int,
        *,
        client: Any = None,
        location: str = ":memory:",
        sparse_encoder: SparseEncoder | None = None,
        snapshot_dir: str | Path = ".cache/snapshots",
        distance: str = "Cosine",
    ) -> None:
        self.collection = collection
        self.dim = dim
        self.distance = distance
        self._location = location
        self._client = client
        self._sparse = sparse_encoder or HashingBM25Encoder()
        self._snapshot_dir = Path(snapshot_dir)
        self._ensured = False

    # --- client / collection lifecycle ------------------------------------

    def _get_client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(location=self._location)
        return self._client

    async def ensure_collection(self) -> None:
        """Create the hybrid collection if it does not already exist."""
        if self._ensured:
            return
        from qdrant_client import models as m

        client = self._get_client()
        if not await client.collection_exists(self.collection):
            await client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE: m.VectorParams(size=self.dim, distance=m.Distance[self.distance.upper()])
                },
                sparse_vectors_config={SPARSE: m.SparseVectorParams(index=m.SparseIndexParams())},
            )
        self._ensured = True

    # --- write path -------------------------------------------------------

    async def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        from qdrant_client import models as m

        await self.ensure_collection()
        client = self._get_client()
        sparse_vecs = self._sparse.encode_documents([embedding_text(c) for c in chunks])
        points: list[Any] = []
        for chunk, (idx, val) in zip(chunks, sparse_vecs, strict=True):
            if chunk.embedding is None:
                raise KnowledgeAgentError(
                    f"chunk {chunk.chunk_id} has no embedding; embed before upsert"
                )
            points.append(
                m.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector={
                        DENSE: chunk.embedding,
                        SPARSE: m.SparseVector(indices=idx, values=val),
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "acl": chunk.acl,
                        "chunk": chunk.model_dump(mode="json"),
                    },
                )
            )
        await client.upsert(collection_name=self.collection, points=points)

    async def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        client = self._get_client()
        await client.delete(
            collection_name=self.collection,
            points_selector=[_point_id(c) for c in chunk_ids],
        )

    # --- filters ----------------------------------------------------------

    def _build_filter(self, filters: dict | None):  # type: ignore[no-untyped-def]
        from qdrant_client import models as m

        filters = dict(filters or {})
        principals = filters.pop("user_principals", None)
        must: list[Any] = []
        for key, value in filters.items():
            payload_key = "doc_id" if key == "doc_id" else f"chunk.metadata.{key}"
            if isinstance(value, list):
                must.append(m.FieldCondition(key=payload_key, match=m.MatchAny(any=value)))
            else:
                must.append(m.FieldCondition(key=payload_key, match=m.MatchValue(value=value)))
        if principals:
            # ACL: principal intersects chunk.acl, OR chunk is public (empty acl).
            acl_filter = m.Filter(
                should=[
                    m.FieldCondition(key="acl", match=m.MatchAny(any=list(principals))),
                    m.IsEmptyCondition(is_empty=m.PayloadField(key="acl")),
                ]
            )
            must.append(acl_filter)
        return m.Filter(must=must) if must else None

    # --- read path --------------------------------------------------------

    def _to_candidate(self, point: Any, rank: int, retriever: str) -> RetrievalCandidate:
        chunk = Chunk.model_validate(point.payload["chunk"])
        return RetrievalCandidate(
            chunk=chunk, score=float(point.score), retriever=retriever, rank=rank
        )

    async def search_dense(
        self, vec: list[float], k: int, filters: dict | None = None
    ) -> list[RetrievalCandidate]:
        await self.ensure_collection()
        client = self._get_client()
        resp = await client.query_points(
            collection_name=self.collection,
            query=vec,
            using=DENSE,
            limit=k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        return [self._to_candidate(p, i, "dense") for i, p in enumerate(resp.points)]

    async def search_sparse(
        self, query: str, k: int, filters: dict | None = None
    ) -> list[RetrievalCandidate]:
        from qdrant_client import models as m

        await self.ensure_collection()
        client = self._get_client()
        idx, val = self._sparse.encode_query(query)
        if not idx:
            return []
        resp = await client.query_points(
            collection_name=self.collection,
            query=m.SparseVector(indices=idx, values=val),
            using=SPARSE,
            limit=k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        return [self._to_candidate(p, i, "sparse_bm25") for i, p in enumerate(resp.points)]

    # --- versioning -------------------------------------------------------

    async def snapshot(self) -> SnapshotRef:
        """Scroll all points to a local JSON file and return a reference."""
        await self.ensure_collection()
        client = self._get_client()
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self.collection}-{uuid.uuid4().hex[:12]}"
        path = self._snapshot_dir / f"{name}.json"
        records: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = await client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in points:
                records.append({"id": p.id, "vector": p.vector, "payload": p.payload})
            if offset is None:
                break
        path.write_text(json.dumps({"collection": self.collection, "dim": self.dim, "points": records}))
        return SnapshotRef(
            collection=self.collection, name=name, location=str(path), count=len(records)
        )

    async def restore(self, ref: SnapshotRef) -> None:
        """Recreate the collection and re-upsert points from a snapshot file."""
        from qdrant_client import models as m

        if not ref.location:
            raise KnowledgeAgentError("snapshot ref has no location to restore from")
        data = json.loads(Path(ref.location).read_text())
        client = self._get_client()
        await client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: m.VectorParams(size=self.dim, distance=m.Distance[self.distance.upper()])
            },
            sparse_vectors_config={SPARSE: m.SparseVectorParams(index=m.SparseIndexParams())},
        )
        self._ensured = True
        points = []
        for rec in data["points"]:
            vector = rec["vector"]
            # normalize sparse vector dict back to SparseVector
            if isinstance(vector, dict) and SPARSE in vector and isinstance(vector[SPARSE], dict):
                sv = vector[SPARSE]
                vector = {
                    DENSE: vector[DENSE],
                    SPARSE: m.SparseVector(indices=sv["indices"], values=sv["values"]),
                }
            points.append(m.PointStruct(id=rec["id"], vector=vector, payload=rec["payload"]))
        if points:
            await client.upsert(collection_name=self.collection, points=points)

    async def count(self) -> int:
        """Number of points in the collection (convenience for tests/CLI)."""
        await self.ensure_collection()
        client = self._get_client()
        res = await client.count(collection_name=self.collection)
        return int(res.count)


__all__ = ["QdrantIndex", "DENSE", "SPARSE"]
