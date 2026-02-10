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
async def test_create_analysis_record(analysis_record: AnalysisRecord) -> None:
    """Test creating analysis record - placeholder for integration test."""
    pytest.skip("Integration test - requires database setup")


@pytest.mark.asyncio
async def test_get_by_symbol() -> None:
    """Test retrieving analysis records by symbol - placeholder for integration test."""
    pytest.skip("Integration test - requires database setup")


@pytest.mark.asyncio
async def test_delete_before() -> None:
    """Test cleanup of old records - placeholder for integration test."""
    pytest.skip("Integration test - requires database setup")
