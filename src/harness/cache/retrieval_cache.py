"""In-memory LRU + TTL retrieval cache (SPEC §6.12 tier 3).

Keyed on ``(query_hash, index_version, filters_hash)``. The short TTL (default
5 min) lets index updates flow through without serving stale results. Bounded
LRU eviction caps memory. Thread-safe.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

V = TypeVar("V")

DEFAULT_TTL_SECONDS = 300.0  # 5 minutes (SPEC §6.12)


def _hash_obj(obj: Any) -> str:
    """Stable hash of an arbitrary JSON-serializable object (order-independent)."""
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def retrieval_cache_key(query: str, index_version: str, filters: dict[str, Any]) -> str:
    """Compose the cache key from (query_hash, index_version, filters_hash)."""
    return f"{_hash_obj(query)}:{index_version}:{_hash_obj(filters)}"


class RetrievalCache(Generic[V]):
    """Bounded LRU cache with per-entry TTL expiry."""

    def __init__(self, *, max_size: int = 1024, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()
        self._clock = time.monotonic

    def get(self, key: str) -> V | None:
        """Return the cached value, or None if absent or expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)  # mark as recently used
            return value

    def put(self, key: str, value: V) -> None:
        """Insert/refresh a value, evicting the least-recently-used if at capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (self._clock() + self._ttl, value)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)  # evict LRU

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "RetrievalCache",
    "retrieval_cache_key",
]
