"""DCI executor protocol (SPEC §15.1, §15.3) — Phase 5B parallel-build surface.

The orchestrator's ``dci_tool`` node calls a :class:`DCIExecutor` to materialise
filesystem-style corpus reads (``corpus_grep`` / ``corpus_read`` / ...) for the
query. Phase 5A (ka-7xv) lands the concrete implementation that fans the query
out over the six §15.1 tools, runs them inside the Phase 3N sandbox, and folds
the results back into :class:`RetrievalCandidate` objects with citations.

By isolating the call surface here we let 5B wire the orchestrator/router
against a stable contract while 5A is built in parallel — same pattern Phase 4
used. Tests inject a stub executor; production wires the real one once 5A lands.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from common.schemas import Query, RetrievalResult


@runtime_checkable
class DCIExecutor(Protocol):
    """Run a query against the DCI tools and return cited candidates.

    The executor is responsible for picking which §15.1 tools to invoke (grep
    vs glob vs read vs neighbors), enforcing per-tool budget tracking (§6.11),
    running them in the sandbox (§6.7), and returning :class:`RetrievalResult`
    so the dci_tool node can merge candidates into the parent evidence set the
    same way ``retrieve`` does. Implementations attach ``retriever='dci'`` on
    every candidate so the orchestrator can audit provenance.
    """

    name: str

    async def run(self, query: Query, k: int) -> RetrievalResult: ...


__all__ = ["DCIExecutor"]
