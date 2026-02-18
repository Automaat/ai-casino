"""Tests for TradeRepository.get_entry_trade()."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.database.models import TradeORM
from src.database.repositories.trade import TradeRepository
from src.strategies.signal import Signal


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy async session."""
    session = MagicMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def sample_trade_orm() -> MagicMock:
    """Create sample trade ORM."""
    orm = MagicMock(spec=TradeORM)
    orm.id = "test-id"
    orm.timestamp = datetime(2024, 1, 15, tzinfo=UTC)
    orm.symbol = "AAPL"
    orm.action = "BUY"
    orm.entry_price = 150.0
    orm.exit_price = None
    orm.shares = 10
    orm.stop_loss_price = 145.0
    orm.confidence = 0.85
    orm.risk_level = "MEDIUM"
    orm.status = "OPEN"
    orm.pnl = None
    orm.pnl_percent = None
    orm.strategy_name = "momentum"
    orm.broker_order_id = None
    orm.is_paper_trade = True
    orm.closed_at = None
    return orm


@pytest.mark.asyncio
async def test_get_entry_trade_finds_open_buy(mock_session, sample_trade_orm) -> None:
    """Test get_entry_trade finds most recent OPEN BUY trade."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = sample_trade_orm
    mock_session.execute.return_value = mock_result

    repo = TradeRepository(mock_session)
    trade = await repo.get_entry_trade("AAPL")

    assert trade is not None
    assert trade.symbol == "AAPL"
    assert trade.action == Signal.BUY
    assert trade.status == "OPEN"
    assert trade.confidence == 0.85
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_entry_trade_returns_none_when_not_found(mock_session) -> None:
    """Test get_entry_trade returns None when no matching trade found."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    repo = TradeRepository(mock_session)
    trade = await repo.get_entry_trade("TSLA")

    assert trade is None
    mock_session.execute.assert_called_once()
