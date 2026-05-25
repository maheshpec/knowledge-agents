"""Query-op stubs filled in Phase 3 (SPEC §7.6.2).

Registered now so configs/components.yaml and the registry can name them; each
raises ``NotImplementedError`` until its phase lands.
"""

from __future__ import annotations

from common.schemas import Query


class HyDEExpander:
    """Hypothetical Document Embeddings: generate a fake answer, embed it. Phase 3."""

    name = "hyde"

    async def transform(self, query: Query) -> Query:
        raise NotImplementedError("HyDEExpander is a Phase 3 stub (SPEC §7.6.2)")


class Decomposer:
    """Break a multi-part query into sub-queries. Phase 3."""

    name = "decomposer"

    async def transform(self, query: Query) -> Query:
        raise NotImplementedError("Decomposer is a Phase 3 stub (SPEC §7.6.2)")


class Stepback:
    """Generate a broader 'step-back' version of the query. Phase 3."""

    name = "stepback"

    async def transform(self, query: Query) -> Query:
        raise NotImplementedError("Stepback is a Phase 3 stub (SPEC §7.6.2)")


__all__ = ["HyDEExpander", "Decomposer", "Stepback"]
