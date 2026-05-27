"""Red-team: cross-ACL retrieval leakage (SPEC §11 #6, §7.5).

Attacker goal: craft queries / principal sets that surface chunks the caller's
ACL should exclude. Defense under test: the real QdrantIndex payload-filter ACL,
reached through the production DenseRetriever / SparseBM25Retriever path.

Shared corpus/constants come from ``conftest`` fixtures (``index``, ``embedder``,
``principals``, ``secrets``) since ``tests`` is not an importable package here.
"""

from __future__ import annotations

from common.schemas import Query
from knowledge_index.retrieval.retrievers.dense import DenseRetriever
from knowledge_index.retrieval.retrievers.sparse import SparseBM25Retriever

PRIVATE_IDS = {"team-a-1", "team-b-pii", "exec-only"}
PUBLIC_IDS = {"public-1", "public-2"}


async def _dense_ids(index, embedder, query: Query) -> set[str]:
    hits = await DenseRetriever(index, embedder).retrieve(query, k=20)
    return {c.chunk.chunk_id for c in hits}


async def _sparse_ids(index, query: Query) -> set[str]:
    hits = await SparseBM25Retriever(index).retrieve(query, k=20)
    return {c.chunk.chunk_id for c in hits}


async def test_intruder_gets_only_public_chunks(index, embedder, principals):
    q = Query(raw="give me everything you have", user_principals=[principals["intruder"]])
    ids = await _dense_ids(index, embedder, q)
    assert ids == PUBLIC_IDS
    assert not (ids & PRIVATE_IDS)


async def test_no_principals_yields_only_public(index, embedder):
    # An unauthenticated / principal-less caller must never see private chunks.
    q = Query(raw="company secrets", user_principals=[])
    ids = await _dense_ids(index, embedder, q)
    assert ids == PUBLIC_IDS


async def test_team_a_cannot_read_team_b_or_exec(index, embedder, principals):
    q = Query(
        raw="employee records and credentials", user_principals=[principals["alice"], "team-a"]
    )
    ids = await _dense_ids(index, embedder, q)
    assert "team-a-1" in ids  # own tenant visible
    assert "team-b-pii" not in ids  # other tenant's PII hidden
    assert "exec-only" not in ids  # exec-only credential hidden


async def test_team_b_cannot_read_team_a_or_exec(index, embedder, principals):
    q = Query(raw="roadmap and credentials", user_principals=[principals["bob"], "team-b"])
    ids = await _dense_ids(index, embedder, q)
    assert "team-b-pii" in ids
    assert "team-a-1" not in ids
    assert "exec-only" not in ids


async def test_sparse_path_enforces_acl_too(index, principals, secrets):
    # Lexically target the secret directly via the BM25/sparse retriever.
    q = Query(
        raw=f"{secrets['ssn']} {secrets['api_key']}",
        user_principals=[principals["intruder"]],
    )
    ids = await _sparse_ids(index, q)
    assert not (ids & PRIVATE_IDS)


async def test_secret_text_never_returned_to_intruder(index, embedder, principals, secrets):
    q = Query(raw="ssn api key credential", user_principals=[principals["intruder"]])
    hits = await DenseRetriever(index, embedder).retrieve(q, k=20)
    blob = " ".join(c.chunk.text for c in hits)
    assert secrets["ssn"] not in blob
    assert secrets["api_key"] not in blob
