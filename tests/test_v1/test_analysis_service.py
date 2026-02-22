"""Tests for AnalysisService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from result import Err, Ok

from src.daemon.state.models import AnalysisRecord
from src.strategies.session import TradingSession
from src.v1.analysis_service import AnalysisService


def _make_db_engine() -> MagicMock:
    engine = MagicMock()
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    engine.session = MagicMock(return_value=session)
    return engine


def _make_workflow_result(symbol: str = "AAPL") -> MagicMock:
    result = MagicMock()
    result.symbol = symbol
    result.trading_session = TradingSession.REGULAR
    result.decision.action.value = "BUY"
    result.decision.confidence = 0.8
    result.decision.reasoning = ["Strong momentum"]
    result.technical.rsi = 45.0
    result.technical.macd_hist = 0.5
    return result


def _make_repo_mock() -> AsyncMock:
    repo = AsyncMock()
    repo.__aenter__ = AsyncMock(return_value=repo)
    repo.__aexit__ = AsyncMock(return_value=False)
    return repo


class TestRecordNoEngine:
    """Tests for record() when database_engine is None."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_ok_none(self) -> None:
        service = AnalysisService(database_engine=None)
        result = await service.record(_make_workflow_result())
        assert isinstance(result, Ok)
        assert result.ok() is None


class TestRecordWithEngine:
    """Tests for record() with a database engine."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_persists_analysis_record(self) -> None:
        service = AnalysisService(database_engine=_make_db_engine())
        repo = _make_repo_mock()

        with patch("src.di.providers.database.create_analysis_repository", return_value=repo):
            result = await service.record(_make_workflow_result())

        assert isinstance(result, Ok)
        record = result.ok()
        assert isinstance(record, AnalysisRecord)
        assert record.symbol == "AAPL"
        assert record.signal == "BUY"
        assert record.confidence == 0.8
        assert record.rsi == 45.0
        assert record.macd_hist == 0.5
        repo.create.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_err_on_failure(self) -> None:
        service = AnalysisService(database_engine=_make_db_engine())
        repo = _make_repo_mock()
        repo.create = AsyncMock(side_effect=RuntimeError("DB down"))

        with patch("src.di.providers.database.create_analysis_repository", return_value=repo):
            result = await service.record(_make_workflow_result())

        assert isinstance(result, Err)
        assert isinstance(result.err_value, RuntimeError)


class TestGetRecentNoEngine:
    """Tests for get_recent() when database_engine is None."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_empty_list(self) -> None:
        service = AnalysisService(database_engine=None)
        result = await service.get_recent()
        assert isinstance(result, Ok)
        assert result.ok() == []


class TestGetRecentWithEngine:
    """Tests for get_recent() with a database engine."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_records(self) -> None:
        service = AnalysisService(database_engine=_make_db_engine())
        repo = _make_repo_mock()
        records = [MagicMock(spec=AnalysisRecord), MagicMock(spec=AnalysisRecord)]
        repo.get_recent = AsyncMock(return_value=records)

        with patch("src.di.providers.database.create_analysis_repository", return_value=repo):
            result = await service.get_recent(limit=10)

        assert isinstance(result, Ok)
        assert result.ok() == records
        repo.get_recent.assert_called_once_with(10)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_err_on_failure(self) -> None:
        service = AnalysisService(database_engine=_make_db_engine())
        repo = _make_repo_mock()
        repo.get_recent = AsyncMock(side_effect=RuntimeError("Connection lost"))

        with patch("src.di.providers.database.create_analysis_repository", return_value=repo):
            result = await service.get_recent()

        assert isinstance(result, Err)
        assert isinstance(result.err_value, RuntimeError)


class TestGetBySymbolNoEngine:
    """Tests for get_by_symbol() when database_engine is None."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_empty_list(self) -> None:
        service = AnalysisService(database_engine=None)
        result = await service.get_by_symbol("AAPL")
        assert isinstance(result, Ok)
        assert result.ok() == []


class TestGetBySymbolWithEngine:
    """Tests for get_by_symbol() with a database engine."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_records_for_symbol(self) -> None:
        service = AnalysisService(database_engine=_make_db_engine())
        repo = _make_repo_mock()
        records = [MagicMock(spec=AnalysisRecord)]
        repo.get_by_symbol = AsyncMock(return_value=records)

        with patch("src.di.providers.database.create_analysis_repository", return_value=repo):
            result = await service.get_by_symbol("TSLA", limit=50)

        assert isinstance(result, Ok)
        assert result.ok() == records
        repo.get_by_symbol.assert_called_once_with("TSLA", 50)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_err_on_failure(self) -> None:
        service = AnalysisService(database_engine=_make_db_engine())
        repo = _make_repo_mock()
        repo.get_by_symbol = AsyncMock(side_effect=RuntimeError("Timeout"))

        with patch("src.di.providers.database.create_analysis_repository", return_value=repo):
            result = await service.get_by_symbol("TSLA")

        assert isinstance(result, Err)
        assert isinstance(result.err_value, RuntimeError)


class TestRepr:
    """Tests for __repr__."""

    @pytest.mark.unit
    def test_repr_with_engine(self) -> None:
        service = AnalysisService(database_engine=_make_db_engine())
        assert repr(service) == "AnalysisService(db=enabled)"

    @pytest.mark.unit
    def test_repr_without_engine(self) -> None:
        service = AnalysisService(database_engine=None)
        assert repr(service) == "AnalysisService(db=disabled)"
