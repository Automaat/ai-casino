"""Thread-safe in-memory TTL cache."""

import threading
import time


class MemoryTTLCache:
    """Thread-safe in-memory cache with per-entry TTL."""

    def __init__(self) -> None:
        """Initialize empty cache with a reentrant lock."""
        self._store: dict[str, tuple[object, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> object | None:
        """Return cached value or None if missing/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: object, expire: int | None = None) -> None:
        """Store value with optional TTL in seconds."""
        expires_at = time.monotonic() + expire if expire is not None else float("inf")
        with self._lock:
            self._store[key] = (value, expires_at)

    def expire(self) -> int:
        """Evict expired entries. Returns count removed."""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired:
                del self._store[k]
            return len(expired)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    def close(self) -> None:
        """No-op — compatibility shim."""

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MemoryTTLCache(entries={len(self._store)})"
