"""Direct Corpus Interaction (DCI) — Phase 5 (SPEC §15).

Filesystem-style tools that let the orchestrator interact with the indexed
corpus directly (grep / glob / ls / read / describe / neighbors) instead of —
or alongside — vector retrieval. Each tool is a sandbox-compatible callable
(SPEC §6.7) that returns citation-bearing results and enforces ACLs against
the caller's principals (SPEC §11 #6).

The public surface:

* :class:`CorpusStore` — protocol the tools call into. The in-memory backing
  store :class:`InMemoryCorpusStore` reassembles docs from indexed chunks;
  prod stores can layer their own implementation (e.g. backed by S3 / Qdrant
  scrolls).
* The six tool classes plus the :func:`make_dci_tools` factory the orchestrator
  uses to register them.
* :class:`DCIExecutor` — the executor protocol (Phase 5B) the orchestrator
  and router are wired against.
* :func:`dci_policy` — the default :class:`SandboxPolicy` for DCI tools
  (no network, read-only FS, modest CPU/memory caps).
"""

from __future__ import annotations

from knowledge_index.dci.base import (
    ChunkRef,
    CorpusStore,
    DirectoryEntry,
    DirectoryListing,
    DocMetadata,
    DocRef,
    DocSlice,
    GrepHit,
    InMemoryCorpusStore,
)
from knowledge_index.dci.protocol import DCIExecutor
from knowledge_index.dci.sandbox import dci_policy
from knowledge_index.dci.tools import (
    CorpusDescribeTool,
    CorpusGlobTool,
    CorpusGrepTool,
    CorpusLsTool,
    CorpusNeighborsTool,
    CorpusReadTool,
    DCITool,
    make_dci_tools,
)

__all__ = [
    # schemas
    "DocRef",
    "DocSlice",
    "DocMetadata",
    "GrepHit",
    "ChunkRef",
    "DirectoryEntry",
    "DirectoryListing",
    # store
    "CorpusStore",
    "InMemoryCorpusStore",
    # tools
    "DCITool",
    "CorpusGrepTool",
    "CorpusGlobTool",
    "CorpusLsTool",
    "CorpusReadTool",
    "CorpusDescribeTool",
    "CorpusNeighborsTool",
    "make_dci_tools",
    # executor protocol (Phase 5B)
    "DCIExecutor",
    # sandbox
    "dci_policy",
]
