"""Shared fixtures for the ACL / PII red-team suite (ka-wig, SPEC §11 #6, §13).

These tests exercise the *real* access-control path, not a fake:

* ``QdrantIndex`` runs in Qdrant's in-memory mode (``location=":memory:"``), so
  the genuine payload-filter ACL enforcement is under test — no external service.
* A deterministic embedder makes dense search reproducible offline.

The corpus deliberately mixes public chunks, several private ACL tenants, and
chunks carrying PII so attacker scenarios have something real to try to steal.
"""

from __future__ import annotations

import pytest

from common.schemas import Chunk
from knowledge_index.indexing.qdrant_index import QdrantIndex

DIM = 4

# Principals
ALICE = "user:alice"  # member of team-a
BOB = "user:bob"  # member of team-b
INTRUDER = "user:mallory"  # member of nothing privileged

# A canary string that must NEVER appear in any output an unauthorized caller sees.
SECRET_SSN = "SSN 123-45-6789"
SECRET_EMAIL = "ceo.private@example.com"
SECRET_API_KEY = "sk-live-DEADBEEFsecrettoken"


class DeterministicEmbedder:
    """Maps each chunk/query to a fixed vector by keyword, so search is stable.

    Every document here shares the same dense vector; that is intentional — it
    means dense recall is maximal and the ONLY thing that can keep a private
    chunk out of an intruder's results is the ACL filter. If the filter regresses,
    these tests surface the leak instead of hiding behind low similarity.
    """

    name = "deterministic"
    dim = DIM

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _chunk(chunk_id: str, text: str, acl: list[str]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        acl=acl,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )


def corpus() -> list[Chunk]:
    """Mixed-ACL corpus: public, team-a private, team-b private (PII), restricted."""
    return [
        _chunk("public-1", "The company was founded in 1999. This is public.", acl=[]),
        _chunk("public-2", "Office hours are 9 to 5. Also public.", acl=[]),
        _chunk("team-a-1", "team-a roadmap: launch project falcon in Q3.", acl=["team-a"]),
        _chunk(
            "team-b-pii",
            f"Employee record. {SECRET_SSN}. Contact {SECRET_EMAIL}.",
            acl=["team-b"],
        ),
        _chunk(
            "exec-only",
            f"Production credential rotation: {SECRET_API_KEY}",
            acl=["role:exec"],
        ),
    ]


@pytest.fixture
async def index() -> QdrantIndex:
    """A populated in-memory QdrantIndex (real ACL enforcement, no network)."""
    idx = QdrantIndex(collection="redteam", dim=DIM, location=":memory:")
    await idx.upsert(corpus())
    return idx


@pytest.fixture
def embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder()


# Shared constants are exposed as fixtures (not importable module attrs) because
# ``tests`` is not a package on sys.path — the repo convention is fixtures-only.


@pytest.fixture
def principals() -> dict[str, str]:
    return {"alice": ALICE, "bob": BOB, "intruder": INTRUDER}


@pytest.fixture
def secrets() -> dict[str, str]:
    return {"ssn": SECRET_SSN, "email": SECRET_EMAIL, "api_key": SECRET_API_KEY}


__all__ = [
    "ALICE",
    "BOB",
    "INTRUDER",
    "SECRET_SSN",
    "SECRET_EMAIL",
    "SECRET_API_KEY",
    "DeterministicEmbedder",
    "corpus",
]
