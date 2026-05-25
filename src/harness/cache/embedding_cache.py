"""SQLite-backed embedding cache (SPEC §6.12 tier 2).

Keyed on ``(model, sha256(text))`` so identical text under the same embedding
model is embedded exactly once. Embeddings are stored as JSON-encoded float
lists. Safe for concurrent readers/writers via SQLite's own locking.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Persistent cache mapping (model, text) -> embedding vector."""

    def __init__(self, path: str | Path = ".cache/embeddings.sqlite") -> None:
        self._path = Path(path)
        if self._path.parent and str(self._path.parent) not in ("", "."):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + our own lock lets the cache be shared across
        # the asyncio executor threads used by embedders.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model      TEXT NOT NULL,
                    text_hash  TEXT NOT NULL,
                    vector     TEXT NOT NULL,
                    PRIMARY KEY (model, text_hash)
                )
                """
            )
            self._conn.commit()

    def get(self, model: str, text: str) -> list[float] | None:
        """Return the cached embedding for (model, text), or None on a miss."""
        with self._lock:
            row = self._conn.execute(
                "SELECT vector FROM embeddings WHERE model = ? AND text_hash = ?",
                (model, _text_hash(text)),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, text: str, vector: list[float]) -> None:
        """Store an embedding for (model, text), overwriting any prior value."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, vector) VALUES (?, ?, ?)",
                (model, _text_hash(text), json.dumps(vector)),
            )
            self._conn.commit()

    def get_many(self, model: str, texts: list[str]) -> dict[str, list[float]]:
        """Batch lookup; returns {text: vector} only for hits."""
        out: dict[str, list[float]] = {}
        for text in texts:
            hit = self.get(model, text)
            if hit is not None:
                out[text] = hit
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["EmbeddingCache"]
