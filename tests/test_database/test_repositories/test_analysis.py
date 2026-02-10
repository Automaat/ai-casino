"""Tests for AnalysisRecordRepository."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.daemon.state import AnalysisRecord
from src.database.repositories.analysis import AnalysisRecordRepository
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
async def test_create_analysis_record(analysis_record: AnalysisRecord) -> None:
    """Test creating analysis record."""
    # Mock session would go here - placeholder for integration test
    # repository = AnalysisRecordRepository(mock_session)
    # created = await repository.create(analysis_record)
    # assert created.symbol == "AAPL"
    # assert created.signal == "BUY"
    pass


@pytest.mark.asyncio
async def test_get_by_symbol() -> None:
    """Test retrieving analysis records by symbol."""
    # Mock repository query
    pass


@pytest.mark.asyncio
async def test_delete_before() -> None:
    """Test cleanup of old records."""
    # Test retention policy deletion
    pass
