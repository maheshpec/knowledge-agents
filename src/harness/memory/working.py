"""Working memory (SPEC §6.3) — the current turn's scratchpad.

In-process dict, cleared every turn. No persistence, no embedding: reads are a
cheap substring/recency scan since the working set is tiny.
"""

from __future__ import annotations

from collections.abc import Callable

from common.types import MemoryItem
from harness.memory.base import MemoryScope


class WorkingMemory:
    """Per-turn, in-process key→item store."""

    scope: MemoryScope = "working"

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    async def write(self, item: MemoryItem) -> None:
        self._items[item.key] = item

    async def read(self, query: str, k: int = 5) -> list[MemoryItem]:
        q = query.lower().strip()
        items = list(self._items.values())
        if q:
            items = [it for it in items if q in it.key.lower() or q in str(it.value).lower()]
        # most-recent first
        items.sort(key=lambda it: it.created_at, reverse=True)
        return items[:k]

    async def forget(self, predicate: Callable[[MemoryItem], bool]) -> None:
        self._items = {k: v for k, v in self._items.items() if not predicate(v)}

    async def all(self) -> list[MemoryItem]:
        return list(self._items.values())

    def clear(self) -> None:
        """Drop everything — called at the end of each turn."""
        self._items.clear()


__all__ = ["WorkingMemory"]
