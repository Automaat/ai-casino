"""Tests for MarketEventQueue service and MarketEventQueueRepository."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.event_queue.models import MarketEventQueueORM, QueuedMarketEvent
from src.event_queue.repository import MarketEventQueueRepository
from src.event_queue.service import MarketEventQueue


def _make_orm_record(
    event_id: str,
    enqueued_at: datetime,
    consumed_at: datetime | None = None,
    hours_until_expiry: int = 4,
) -> MarketEventQueueORM:
    record = MarketEventQueueORM()
    record.id = uuid.uuid4()
    record.event_id = event_id
    record.event_type = "news"
    record.payload = {"event": {}, "triage": {}}
    record.enqueued_at = enqueued_at
    record.expires_at = enqueued_at + timedelta(hours=hours_until_expiry)
    record.consumed_at = consumed_at
    return record


def _make_queued(event_id: str, enqueued_at: datetime) -> QueuedMarketEvent:
    return QueuedMarketEvent(
        event_id=event_id,
        event_type="news",
        payload={},
        enqueued_at=enqueued_at,
    )


@pytest.fixture
def mock_db_engine():
    """Mock DatabaseEngine with session context manager."""
    session = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)
    engine = MagicMock()
    engine.session.return_value = cm
    return engine, session


@pytest.mark.unit
class TestEnqueueValidation:
    """enqueue rejects invalid ttl_hours."""

    @pytest.mark.asyncio
    async def test_zero_ttl_raises(self, mock_db_engine):
        engine, _ = mock_db_engine
        svc = MarketEventQueue(engine)
        event = MagicMock(event_id="e1", event_type="news")
        triage = MagicMock()
        with pytest.raises(ValueError, match="ttl_hours"):
            await svc.enqueue(event, triage, ttl_hours=0)

    @pytest.mark.asyncio
    async def test_negative_ttl_raises(self, mock_db_engine):
        engine, _ = mock_db_engine
        svc = MarketEventQueue(engine)
        event = MagicMock(event_id="e1", event_type="news")
        triage = MagicMock()
        with pytest.raises(ValueError, match="ttl_hours"):
            await svc.enqueue(event, triage, ttl_hours=-1)


@pytest.mark.unit
class TestDequeueValidation:
    """dequeue rejects invalid max_items."""

    @pytest.mark.asyncio
    async def test_zero_max_items_raises(self, mock_db_engine):
        engine, _ = mock_db_engine
        svc = MarketEventQueue(engine)
        with pytest.raises(ValueError, match="max_items"):
            await svc.dequeue(max_items=0)

    @pytest.mark.asyncio
    async def test_negative_max_items_raises(self, mock_db_engine):
        engine, _ = mock_db_engine
        svc = MarketEventQueue(engine)
        with pytest.raises(ValueError, match="max_items"):
            await svc.dequeue(max_items=-5)


@pytest.mark.unit
class TestMarketEventQueueService:
    """Service delegates correctly to repository."""

    @pytest.mark.asyncio
    async def test_enqueue_sets_correct_expiry(self, mock_db_engine):
        """enqueue record has expires_at = enqueued_at + ttl_hours."""
        engine, _ = mock_db_engine
        captured: list[MarketEventQueueORM] = []

        async def capture_enqueue(_self, record: MarketEventQueueORM) -> None:
            captured.append(record)

        with patch.object(MarketEventQueueRepository, "enqueue", new=capture_enqueue):
            svc = MarketEventQueue(engine)
            event = MagicMock(event_id="e1", event_type="news")
            event.model_dump.return_value = {}
            triage = MagicMock()
            triage.model_dump.return_value = {}
            await svc.enqueue(event, triage, ttl_hours=2)

        assert len(captured) == 1
        record = captured[0]
        assert record.event_id == "e1"
        delta = record.expires_at - record.enqueued_at
        assert abs(delta.total_seconds() - 7200) < 2

    @pytest.mark.asyncio
    async def test_dequeue_returns_repo_results(self, mock_db_engine):
        """dequeue returns what the repository returns."""
        engine, _ = mock_db_engine
        now = datetime.now(UTC)
        expected = [_make_queued("e1", now)]

        async def fake_dequeue(_self, max_items: int = 1) -> list[QueuedMarketEvent]:
            return expected

        with patch.object(MarketEventQueueRepository, "dequeue", new=fake_dequeue):
            svc = MarketEventQueue(engine)
            result = await svc.dequeue(max_items=1)

        assert result == expected

    @pytest.mark.asyncio
    async def test_purge_expired_returns_count(self, mock_db_engine):
        engine, _ = mock_db_engine

        async def fake_purge(_self) -> int:
            return 3

        with patch.object(MarketEventQueueRepository, "purge_expired", new=fake_purge):
            svc = MarketEventQueue(engine)
            result = await svc.purge_expired()

        assert result == 3

    @pytest.mark.asyncio
    async def test_size_returns_pending_count(self, mock_db_engine):
        engine, _ = mock_db_engine

        async def fake_count(_self) -> int:
            return 5

        with patch.object(MarketEventQueueRepository, "count_pending", new=fake_count):
            svc = MarketEventQueue(engine)
            result = await svc.size()

        assert result == 5

    def test_repr_contains_class_name(self, mock_db_engine):
        engine, _ = mock_db_engine
        svc = MarketEventQueue(engine)
        assert "MarketEventQueue" in repr(svc)


@pytest.mark.unit
class TestProcessAfterEnqueue:
    """enqueue stores process_after correctly."""

    @pytest.mark.asyncio
    async def test_enqueue_stores_process_after(self, mock_db_engine):
        """process_after is propagated to the ORM record."""
        engine, _ = mock_db_engine
        captured: list[MarketEventQueueORM] = []
        future = datetime.now(UTC) + timedelta(hours=1)

        async def capture_enqueue(_self, record: MarketEventQueueORM) -> None:
            captured.append(record)

        with patch.object(MarketEventQueueRepository, "enqueue", new=capture_enqueue):
            svc = MarketEventQueue(engine)
            event = MagicMock(event_id="e_pa", event_type="news")
            event.model_dump.return_value = {}
            triage = MagicMock()
            triage.model_dump.return_value = {}
            await svc.enqueue(event, triage, process_after=future)

        assert len(captured) == 1
        assert captured[0].process_after == future

    @pytest.mark.asyncio
    async def test_enqueue_none_process_after_stored(self, mock_db_engine):
        """process_after=None is stored as None (immediately eligible)."""
        engine, _ = mock_db_engine
        captured: list[MarketEventQueueORM] = []

        async def capture_enqueue(_self, record: MarketEventQueueORM) -> None:
            captured.append(record)

        with patch.object(MarketEventQueueRepository, "enqueue", new=capture_enqueue):
            svc = MarketEventQueue(engine)
            event = MagicMock(event_id="e_none", event_type="news")
            event.model_dump.return_value = {}
            triage = MagicMock()
            triage.model_dump.return_value = {}
            await svc.enqueue(event, triage, process_after=None)

        assert captured[0].process_after is None


@pytest.mark.unit
class TestMarketEventQueueRepository:
    """Repository SQL behavior tests."""

    @pytest.mark.asyncio
    async def test_enqueue_uses_on_conflict_do_nothing(self):
        """enqueue executes INSERT ... ON CONFLICT DO NOTHING."""
        session = AsyncMock()
        repo = MarketEventQueueRepository(session)
        now = datetime.now(UTC)
        record = _make_orm_record("e1", now)

        await repo.enqueue(record)

        session.execute.assert_called_once()
        stmt = session.execute.call_args[0][0]
        assert "ON CONFLICT" in str(stmt)
        assert "DO NOTHING" in str(stmt)

    @pytest.mark.asyncio
    async def test_enqueue_idempotent_two_calls(self):
        """Calling enqueue twice executes two statements; idempotency from SQL."""
        session = AsyncMock()
        repo = MarketEventQueueRepository(session)
        now = datetime.now(UTC)
        record = _make_orm_record("e1", now)

        await repo.enqueue(record)
        await repo.enqueue(record)

        assert session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_dequeue_final_select_has_order_by(self):
        """dequeue final SELECT has ORDER BY enqueued_at ASC to guarantee FIFO."""
        from sqlalchemy.dialects import sqlite as sqlite_dialect

        now = datetime.now(UTC)
        ids_result = MagicMock()
        ids_result.fetchall.return_value = [("e1",), ("e2",)]

        orm1 = _make_orm_record("e1", now - timedelta(minutes=5))
        orm2 = _make_orm_record("e2", now)
        rows_result = MagicMock()
        rows_result.scalars.return_value.all.return_value = [orm1, orm2]

        executed_stmts: list = []

        async def capture_execute(stmt, *args, **kwargs):
            executed_stmts.append(stmt)
            if len(executed_stmts) == 1:
                return ids_result
            if len(executed_stmts) == 2:
                return MagicMock()
            return rows_result

        session = AsyncMock()
        session.execute = capture_execute
        session.commit = AsyncMock()

        repo = MarketEventQueueRepository(session)
        await repo.dequeue(max_items=2)

        assert len(executed_stmts) == 3, "expected subquery + update + final select"

        final_sql = str(executed_stmts[2].compile(dialect=sqlite_dialect.dialect()))
        assert "ORDER BY" in final_sql
        assert "enqueued_at" in final_sql.lower()

    @pytest.mark.asyncio
    async def test_dequeue_returns_empty_when_no_pending(self):
        """dequeue returns [] when no pending non-expired events exist."""
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []

        session = AsyncMock()
        session.execute = AsyncMock(return_value=result_mock)
        session.commit = AsyncMock()

        repo = MarketEventQueueRepository(session)
        result = await repo.dequeue(max_items=5)

        assert result == []
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_purge_expired_returns_deleted_count(self):
        """purge_expired returns count of deleted rows."""
        rows = [("id1",), ("id2",), ("id3",)]
        result_mock = MagicMock()
        result_mock.fetchall.return_value = rows

        session = AsyncMock()
        session.execute = AsyncMock(return_value=result_mock)
        session.commit = AsyncMock()

        repo = MarketEventQueueRepository(session)
        count = await repo.purge_expired()

        assert count == 3
        stmt = session.execute.call_args[0][0]
        assert "expires_at" in str(stmt)

    @pytest.mark.asyncio
    async def test_count_pending_returns_scalar(self):
        """count_pending returns scalar from DB."""
        result_mock = MagicMock()
        result_mock.scalar_one.return_value = 7

        session = AsyncMock()
        session.execute = AsyncMock(return_value=result_mock)

        repo = MarketEventQueueRepository(session)
        count = await repo.count_pending()

        assert count == 7

    @pytest.mark.asyncio
    async def test_dequeue_marks_events_consumed(self):
        """dequeue executes UPDATE to set consumed_at on claimed rows."""
        from sqlalchemy.dialects import sqlite as sqlite_dialect

        now = datetime.now(UTC)
        ids_result = MagicMock()
        ids_result.fetchall.return_value = [("e1",)]

        orm1 = _make_orm_record("e1", now - timedelta(minutes=1))
        rows_result = MagicMock()
        rows_result.scalars.return_value.all.return_value = [orm1]

        executed_stmts: list = []

        async def capture_execute(stmt, *args, **kwargs):
            executed_stmts.append(stmt)
            if len(executed_stmts) == 1:
                return ids_result
            if len(executed_stmts) == 2:
                return MagicMock()
            return rows_result

        session = AsyncMock()
        session.execute = capture_execute
        session.commit = AsyncMock()

        repo = MarketEventQueueRepository(session)
        await repo.dequeue(max_items=1)

        assert len(executed_stmts) == 3
        update_sql = str(executed_stmts[1].compile(dialect=sqlite_dialect.dialect()))
        assert "consumed_at" in update_sql.lower()

    @pytest.mark.asyncio
    async def test_dequeue_subquery_filters_process_after(self):
        """dequeue subquery includes process_after <= now filter."""
        from sqlalchemy.dialects import sqlite as sqlite_dialect

        ids_result = MagicMock()
        ids_result.fetchall.return_value = []

        executed_stmts: list = []

        async def capture_execute(stmt, *args, **kwargs):
            executed_stmts.append(stmt)
            return ids_result

        session = AsyncMock()
        session.execute = capture_execute
        session.commit = AsyncMock()

        repo = MarketEventQueueRepository(session)
        await repo.dequeue(max_items=1)

        assert len(executed_stmts) == 1
        sql = str(executed_stmts[0].compile(dialect=sqlite_dialect.dialect()))
        assert "process_after" in sql.lower()

    @pytest.mark.asyncio
    async def test_count_pending_filters_process_after(self):
        """count_pending query includes process_after filter."""
        from sqlalchemy.dialects import sqlite as sqlite_dialect

        result_mock = MagicMock()
        result_mock.scalar_one.return_value = 2

        executed_stmts: list = []

        async def capture_execute(stmt, *args, **kwargs):
            executed_stmts.append(stmt)
            return result_mock

        session = AsyncMock()
        session.execute = capture_execute

        repo = MarketEventQueueRepository(session)
        await repo.count_pending()

        assert len(executed_stmts) == 1
        sql = str(executed_stmts[0].compile(dialect=sqlite_dialect.dialect()))
        assert "process_after" in sql.lower()
