"""Tests for AnalysisRecordRepository."""

from datetime import UTC, datetime

import pytest

from src.daemon.state import AnalysisRecord
from src.strategies.session import TradingSession


@pytest.fixture
def analysis_record() -> AnalysisRecord:
    """Create sample analysis record."""
    return AnalysisRecord(
        symbol="AAPL",
        timestamp=datetime.now(UTC),
        signal="BUY",
        confidence=0.85,
        executed_trade=True,
        trading_session=TradingSession.REGULAR,
        is_paper_trade=True,
        rsi=45.5,
        macd_hist=0.25,
        reasoning=["Strong bullish momentum", "RSI oversold"],
    )


@pytest.mark.asyncio
async def test_create_analysis_record(async_session, analysis_record: AnalysisRecord) -> None:
    """Test creating analysis record."""
    from src.database.repositories.analysis import AnalysisRecordRepository

    repo = AnalysisRecordRepository(async_session)
    result = await repo.create(analysis_record)

    assert result.symbol == analysis_record.symbol
    assert result.signal == analysis_record.signal
    assert result.confidence == analysis_record.confidence
    assert result.executed_trade is True


@pytest.mark.asyncio
async def test_get_by_symbol(async_session, analysis_record: AnalysisRecord) -> None:
    """Test retrieving analysis records by symbol."""
    from src.database.repositories.analysis import AnalysisRecordRepository

    repo = AnalysisRecordRepository(async_session)
    await repo.create(analysis_record)

    results = await repo.get_by_symbol("AAPL")

    assert len(results) == 1
    assert results[0].symbol == "AAPL"
    assert results[0].signal == "BUY"


@pytest.mark.asyncio
async def test_delete_before(async_session, analysis_record: AnalysisRecord) -> None:
    """Test cleanup of old records."""
    from datetime import timedelta

    from src.database.repositories.analysis import AnalysisRecordRepository

    repo = AnalysisRecordRepository(async_session)
    await repo.create(analysis_record)

    cutoff = datetime.now(UTC) + timedelta(days=1)
    deleted_count = await repo.delete_before(cutoff)

    assert deleted_count == 1


@pytest.mark.asyncio
async def test_get_last_analysis_timestamps_empty(async_session) -> None:
    """Test get_last_analysis_timestamps with empty input returns empty dict."""
    from src.database.repositories.analysis import AnalysisRecordRepository

    repo = AnalysisRecordRepository(async_session)
    result = await repo.get_last_analysis_timestamps([])

    assert result == {}


@pytest.mark.asyncio
async def test_get_last_analysis_timestamps_single(async_session, analysis_record: AnalysisRecord) -> None:
    """Test get_last_analysis_timestamps returns most recent timestamp per symbol."""
    from src.database.repositories.analysis import AnalysisRecordRepository

    repo = AnalysisRecordRepository(async_session)
    await repo.create(analysis_record)

    result = await repo.get_last_analysis_timestamps(["AAPL"])

    assert "AAPL" in result
    assert isinstance(result["AAPL"], datetime)


@pytest.mark.asyncio
async def test_get_last_analysis_timestamps_multiple_records(
    async_session, analysis_record: AnalysisRecord
) -> None:
    """Test get_last_analysis_timestamps returns latest when multiple records exist."""
    from datetime import timedelta

    from src.database.repositories.analysis import AnalysisRecordRepository

    repo = AnalysisRecordRepository(async_session)
    older_record = AnalysisRecord(
        symbol="AAPL",
        timestamp=datetime.now(UTC) - timedelta(hours=2),
        signal="SELL",
        confidence=0.6,
        executed_trade=False,
        trading_session=analysis_record.trading_session,
        is_paper_trade=True,
        rsi=75.0,
        macd_hist=-0.1,
        reasoning=["Overbought"],
    )
    await repo.create(older_record)
    await repo.create(analysis_record)

    result = await repo.get_last_analysis_timestamps(["AAPL"])

    assert "AAPL" in result
    # SQLite returns naive datetimes; compare as naive UTC
    returned_ts = result["AAPL"]
    older_ts = older_record.timestamp.replace(tzinfo=None)
    assert returned_ts >= older_ts


@pytest.mark.asyncio
async def test_get_last_analysis_timestamps_missing_symbol(
    async_session, analysis_record: AnalysisRecord
) -> None:
    """Test get_last_analysis_timestamps omits symbols with no records."""
    from src.database.repositories.analysis import AnalysisRecordRepository

    repo = AnalysisRecordRepository(async_session)
    await repo.create(analysis_record)

    result = await repo.get_last_analysis_timestamps(["AAPL", "TSLA"])

    assert "AAPL" in result
    assert "TSLA" not in result
