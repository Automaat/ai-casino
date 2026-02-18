"""In-memory TTL cache for LLM responses."""

import hashlib
import time

from loguru import logger
from pydantic import BaseModel


class LLMResponseCache:
    """In-memory TTL cache for LLM responses.

    Avoids redundant LLM calls when watchers re-trigger analyses
    for the same symbol within a short window.
    """

    def __init__(self, ttl_seconds: int = 900, max_entries: int = 500) -> None:
        """Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cached entries in seconds
            max_entries: Maximum number of cached entries (evicts oldest on overflow)
        """
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._store: dict[str, tuple[float, str | BaseModel]] = {}

    def get(self, key: str) -> str | BaseModel | None:
        """Get cached value if present and not expired.

        Args:
            key: Cache key (from make_key)

        Returns:
            Cached value or None if miss/expired
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        timestamp, value = entry
        if time.monotonic() - timestamp > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str | BaseModel) -> None:
        """Store value in cache.

        Args:
            key: Cache key (from make_key)
            value: Value to cache
        """
        if len(self._store) >= self._max_entries:
            self._evict_oldest()
        self._store[key] = (time.monotonic(), value)

    def _evict_oldest(self) -> None:
        """Evict oldest entry by timestamp."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k][0])
        del self._store[oldest_key]

    @staticmethod
    def make_key(method: str, prompt: str, model: str, temperature: float) -> str:
        """Create deterministic cache key from call parameters.

        Args:
            method: LLM method name (acomplete, astructured)
            prompt: Full prompt text
            model: Model identifier
            temperature: Temperature setting

        Returns:
            SHA-256 hex digest
        """
        raw = f"{method}:{model}:{temperature}:{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._store)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()
        logger.debug("LLM response cache cleared")
