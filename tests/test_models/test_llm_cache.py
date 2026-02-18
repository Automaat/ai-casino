"""Tests for LLMResponseCache."""

import asyncio
import time

import pytest

from src.models.llm_cache import LLMResponseCache


@pytest.mark.unit
class TestMakeKey:
    def test_basic_key(self):
        key = LLMResponseCache.make_key("acomplete", "hello", "gpt-4o", 0.7)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex

    def test_different_methods_differ(self):
        k1 = LLMResponseCache.make_key("acomplete", "p", "m", 0.5)
        k2 = LLMResponseCache.make_key("astructured", "p", "m", 0.5)
        assert k1 != k2

    def test_different_models_differ(self):
        k1 = LLMResponseCache.make_key("acomplete", "p", "gpt-4o", 0.5)
        k2 = LLMResponseCache.make_key("acomplete", "p", "claude-3", 0.5)
        assert k1 != k2

    def test_different_temperatures_differ(self):
        k1 = LLMResponseCache.make_key("acomplete", "p", "m", 0.3)
        k2 = LLMResponseCache.make_key("acomplete", "p", "m", 0.7)
        assert k1 != k2

    def test_different_system_prompts_differ(self):
        k1 = LLMResponseCache.make_key("acomplete", "p", "m", 0.7, system="You are helpful")
        k2 = LLMResponseCache.make_key("acomplete", "p", "m", 0.7, system="You are strict")
        assert k1 != k2

    def test_none_vs_empty_system_equivalent(self):
        k1 = LLMResponseCache.make_key("acomplete", "p", "m", 0.7, system=None)
        k2 = LLMResponseCache.make_key("acomplete", "p", "m", 0.7, system="")
        assert k1 == k2

    def test_different_max_tokens_differ(self):
        k1 = LLMResponseCache.make_key("acomplete", "p", "m", 0.7, max_tokens=100)
        k2 = LLMResponseCache.make_key("acomplete", "p", "m", 0.7, max_tokens=500)
        assert k1 != k2

    def test_none_max_tokens_differs_from_value(self):
        k1 = LLMResponseCache.make_key("acomplete", "p", "m", 0.7, max_tokens=None)
        k2 = LLMResponseCache.make_key("acomplete", "p", "m", 0.7, max_tokens=100)
        assert k1 != k2

    def test_deterministic(self):
        k1 = LLMResponseCache.make_key("acomplete", "prompt", "model", 0.5, "sys", 256)
        k2 = LLMResponseCache.make_key("acomplete", "prompt", "model", 0.5, "sys", 256)
        assert k1 == k2


@pytest.mark.unit
class TestCacheGetSet:
    @pytest.mark.asyncio
    async def test_miss_returns_none(self):
        cache = LLMResponseCache()
        assert await cache.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_then_get(self):
        cache = LLMResponseCache()
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        cache = LLMResponseCache(ttl_seconds=1)
        await cache.set("key1", "value")
        assert await cache.get("key1") == "value"
        # Manually expire by manipulating timestamp
        cache._store["key1"] = (time.monotonic() - 2, "value")
        assert await cache.get("key1") is None
        assert "key1" not in cache._store

    @pytest.mark.asyncio
    async def test_size_property(self):
        cache = LLMResponseCache()
        assert cache.size == 0
        await cache.set("k1", "v1")
        assert cache.size == 1
        await cache.set("k2", "v2")
        assert cache.size == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        cache = LLMResponseCache()
        await cache.set("k1", "v1")
        await cache.set("k2", "v2")
        await cache.clear()
        assert cache.size == 0
        assert await cache.get("k1") is None


@pytest.mark.unit
class TestMaxEntries:
    @pytest.mark.asyncio
    async def test_evicts_oldest_on_overflow(self):
        cache = LLMResponseCache(ttl_seconds=3600, max_entries=3)
        await cache.set("k1", "v1")
        await cache.set("k2", "v2")
        await cache.set("k3", "v3")
        assert cache.size == 3
        # Insert 4th entry — oldest (k1) should be evicted
        await cache.set("k4", "v4")
        assert cache.size == 3
        assert await cache.get("k1") is None
        assert await cache.get("k4") == "v4"


@pytest.mark.unit
class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_writes_no_race(self):
        cache = LLMResponseCache(ttl_seconds=3600, max_entries=1000)

        async def write(i: int) -> None:
            await cache.set(f"key_{i}", f"value_{i}")

        await asyncio.gather(*[write(i) for i in range(50)])
        assert cache.size == 50

    @pytest.mark.asyncio
    async def test_concurrent_reads_no_error(self):
        cache = LLMResponseCache()
        await cache.set("shared", "data")

        async def read() -> str | None:
            return await cache.get("shared")

        results = await asyncio.gather(*[read() for _ in range(20)])
        assert all(r == "data" for r in results)
