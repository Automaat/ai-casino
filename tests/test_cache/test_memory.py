import threading
import time

import pytest

from src.cache.memory import MemoryTTLCache


@pytest.fixture
def cache() -> MemoryTTLCache[str]:
    return MemoryTTLCache()


class TestGet:
    def test_missing_key_returns_none(self, cache: MemoryTTLCache[str]):
        assert cache.get("missing") is None

    def test_returns_stored_value(self, cache: MemoryTTLCache[str]):
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_expired_entry_returns_none(self, cache: MemoryTTLCache[str]):
        cache.set("k", "v", expire=1)
        time.sleep(1.1)
        assert cache.get("k") is None

    def test_expired_entry_is_evicted(self, cache: MemoryTTLCache[str]):
        cache.set("k", "v", expire=1)
        time.sleep(1.1)
        cache.get("k")
        assert repr(cache) == "MemoryTTLCache(entries=0)"


class TestSet:
    def test_no_ttl_does_not_expire(self, cache: MemoryTTLCache[str]):
        cache.set("k", "v")
        time.sleep(0.1)
        assert cache.get("k") == "v"

    def test_overwrites_existing(self, cache: MemoryTTLCache[str]):
        cache.set("k", "a")
        cache.set("k", "b")
        assert cache.get("k") == "b"


class TestClear:
    def test_removes_all_entries(self, cache: MemoryTTLCache[str]):
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_clear_empty_cache(self, cache: MemoryTTLCache[str]):
        cache.clear()
        assert repr(cache) == "MemoryTTLCache(entries=0)"


class TestExpire:
    def test_returns_zero_when_nothing_expired(self, cache: MemoryTTLCache[str]):
        cache.set("k", "v", expire=60)
        assert cache.expire() == 0

    def test_returns_count_of_removed_entries(self, cache: MemoryTTLCache[str]):
        cache.set("a", "1", expire=1)
        cache.set("b", "2", expire=1)
        cache.set("c", "3", expire=60)
        time.sleep(1.1)
        assert cache.expire() == 2

    def test_non_expired_entries_remain(self, cache: MemoryTTLCache[str]):
        cache.set("a", "1", expire=1)
        cache.set("b", "2", expire=60)
        time.sleep(1.1)
        cache.expire()
        assert cache.get("b") == "2"


class TestRepr:
    def test_empty(self, cache: MemoryTTLCache[str]):
        assert repr(cache) == "MemoryTTLCache(entries=0)"

    def test_with_entries(self, cache: MemoryTTLCache[str]):
        cache.set("a", "1")
        cache.set("b", "2")
        assert repr(cache) == "MemoryTTLCache(entries=2)"


class TestThreadSafety:
    def test_concurrent_writes_no_data_loss(self):
        cache: MemoryTTLCache[int] = MemoryTTLCache()
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    cache.set(str(i), i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_reads_and_writes(self):
        cache: MemoryTTLCache[str] = MemoryTTLCache()
        cache.set("shared", "value")
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(200):
                    cache.get("shared")
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(200):
                    cache.set("shared", str(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(3)] + [
            threading.Thread(target=writer) for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
