"""Memory contracts (SPEC §6.3).

Three scopes — ``working`` (per-turn scratch), ``session`` (recent turns + facts),
``long_term`` (durable user/project facts in a vector index) — sit behind one
:class:`Memory` protocol. Each scope is a :class:`ScopeStore`; :class:`LayeredMemory`
(see ``manager.py``) routes ``write``/``read``/``forget`` to the right store.

``MemoryItem`` already lives in :mod:`common.types`; it is the unit every store
reads and writes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

from common.types import MemoryItem

MemoryScope = Literal["working", "session", "long_term"]


@runtime_checkable
class Memory(Protocol):
    """Scope-aware memory facade (SPEC §6.3)."""

    async def write(self, key: str, value: Any, scope: MemoryScope) -> None: ...
    async def read(self, query: str, scope: MemoryScope, k: int = 5) -> list[MemoryItem]: ...
    async def forget(self, predicate: Callable[[MemoryItem], bool]) -> None: ...


@runtime_checkable
class ScopeStore(Protocol):
    """One memory scope's backing store."""

    scope: MemoryScope

    async def write(self, item: MemoryItem) -> None: ...
    async def read(self, query: str, k: int = 5) -> list[MemoryItem]: ...
    async def forget(self, predicate: Callable[[MemoryItem], bool]) -> None: ...
    async def all(self) -> list[MemoryItem]: ...


__all__ = ["MemoryScope", "Memory", "ScopeStore"]
