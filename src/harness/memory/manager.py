"""Layered memory facade (SPEC §6.3).

Routes the :class:`Memory` protocol across the three scope stores and adds
``consolidate`` — the extraction-gated path that long-term writes should use
(SPEC §6.3: "do not naively store everything").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from common.types import MemoryItem
from harness.memory.base import MemoryScope, ScopeStore
from harness.memory.extraction import MemoryExtractor


class LayeredMemory:
    """Working + session + long-term memory behind one facade (SPEC §6.3)."""

    def __init__(
        self,
        *,
        working: ScopeStore,
        session: ScopeStore,
        long_term: ScopeStore,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        self._stores: dict[MemoryScope, ScopeStore] = {
            "working": working,
            "session": session,
            "long_term": long_term,
        }
        self._extractor = extractor

    def store(self, scope: MemoryScope) -> ScopeStore:
        return self._stores[scope]

    async def write(self, key: str, value: Any, scope: MemoryScope) -> None:
        await self._stores[scope].write(MemoryItem(key=key, value=value, scope=scope))

    async def read(self, query: str, scope: MemoryScope, k: int = 5) -> list[MemoryItem]:
        return await self._stores[scope].read(query, k)

    async def forget(self, predicate: Callable[[MemoryItem], bool]) -> None:
        # Protocol forget is scope-agnostic: apply across every scope.
        for store in self._stores.values():
            await store.forget(predicate)

    async def consolidate(self, text: str, *, scope: MemoryScope = "long_term") -> list[MemoryItem]:
        """Extract durable facts from ``text`` and write them to ``scope``.

        This is the gated write path (SPEC §6.3) — the extractor decides what is
        worth keeping; nothing is stored when it returns no facts. Returns the
        items that were written.
        """
        if self._extractor is None:
            raise ValueError("consolidate requires a MemoryExtractor")
        items = await self._extractor.extract(text, scope=scope)
        store = self._stores[scope]
        for item in items:
            await store.write(item)
        return items


__all__ = ["LayeredMemory"]
