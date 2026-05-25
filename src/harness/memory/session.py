"""Session memory (SPEC §6.3) — last N turns + extracted facts, in SQLite.

Session-scoped durable store that complements the LangGraph checkpointer (which
persists graph *state*); this holds the distilled conversational facts a session
should carry forward. Keyed on ``(session_id, key)`` so re-writing a key updates
it. Reads return the most-recent matching items.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.types import MemoryItem


class SessionMemory:
    """SQLite-backed, session-scoped memory store."""

    scope = "session"

    def __init__(
        self,
        session_id: str,
        *,
        path: str | Path = ".cache/session_memory.sqlite",
        max_items: int = 100,
    ) -> None:
        self.session_id = session_id
        self.max_items = max_items
        self._path = Path(path)
        if (
            str(self._path) != ":memory:"
            and self._path.parent
            and str(self._path.parent)
            not in (
                "",
                ".",
            )
        ):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_memory (
                    session_id TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    item_json  TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, key)
                )
                """
            )
            self._conn.commit()

    async def write(self, item: MemoryItem) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO session_memory (session_id, key, item_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (self.session_id, item.key, item.model_dump_json(), item.created_at.isoformat()),
            )
            self._conn.commit()

    def _rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT item_json FROM session_memory WHERE session_id = ? "
                "ORDER BY created_at DESC",
                (self.session_id,),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    async def read(self, query: str, k: int = 5) -> list[MemoryItem]:
        q = query.lower().strip()
        items = [MemoryItem.model_validate(d) for d in self._rows()]
        if q:
            items = [it for it in items if q in it.key.lower() or q in str(it.value).lower()]
        return items[:k]

    async def all(self) -> list[MemoryItem]:
        return [MemoryItem.model_validate(d) for d in self._rows()]

    async def forget(self, predicate: Callable[[MemoryItem], bool]) -> None:
        to_drop = [it.key for it in await self.all() if predicate(it)]
        if not to_drop:
            return
        with self._lock:
            self._conn.executemany(
                "DELETE FROM session_memory WHERE session_id = ? AND key = ?",
                [(self.session_id, key) for key in to_drop],
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = ["SessionMemory"]
