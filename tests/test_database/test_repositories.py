"""Tests for database repositories."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.database.repositories.snapshot import PortfolioSnapshot, PortfolioSnapshotRepository
from src.database.repositories.trade import TradeRepository


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy async session."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    return session


class TestTradeRepository:
    """Tests for TradeRepository."""

    @pytest.mark.asyncio
    async def test_create_trade(self, mock_session, sample_trade_record):
        """Test creating a trade record."""
        repo = TradeRepository(mock_session)
        result = await repo.create(sample_trade_record)

        assert mock_session.add.called
        assert mock_session.commit.called
        assert result.symbol == sample_trade_record.symbol

    @pytest.mark.asyncio
    async def test_get_open_trades(self, mock_session, sample_trade_record):
        """Test getting open trades."""
        from src.database.models import TradeORM

        orm = MagicMock(spec=TradeORM)
        orm.id = "test-id"
        orm.timestamp = sample_trade_record.timestamp
        orm.symbol = sample_trade_record.symbol
        orm.action = sample_trade_record.action.value
        orm.entry_price = sample_trade_record.entry_price
        orm.exit_price = None
        orm.shares = sample_trade_record.shares
        orm.stop_loss_price = sample_trade_record.stop_loss_price
        orm.confidence = sample_trade_record.confidence
        orm.risk_level = sample_trade_record.risk_level
        orm.status = "OPEN"
        orm.pnl = None
        orm.pnl_percent = None
        orm.strategy_name = sample_trade_record.strategy_name

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [orm]
        mock_session.execute.return_value = mock_result

        repo = TradeRepository(mock_session)
        trades = await repo.get_open_trades()

        assert len(trades) == 1
        assert trades[0].status == "OPEN"

    def test_repr(self, mock_session):
        """Test string representation."""
        repo = TradeRepository(mock_session)
        assert repr(repo) == "TradeRepository()"


class TestPortfolioSnapshotRepository:
    """Tests for PortfolioSnapshotRepository."""

    @pytest.mark.asyncio
    async def test_create_snapshot(self, mock_session):
        """Test creating a portfolio snapshot."""
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(UTC),
            balance=100000.0,
            available_cash=80000.0,
            total_exposure=20000.0,
            portfolio_value=100000.0,
            positions={"AAPL": 10.0},
            trigger="TRADE",
        )

        repo = PortfolioSnapshotRepository(mock_session)
        result = await repo.create(snapshot)

        assert mock_session.add.called
        assert mock_session.commit.called
        assert result.id is not None

    @pytest.mark.asyncio
    async def test_get_latest(self, mock_session):
        """Test getting latest snapshot."""
        from src.database.models import PortfolioSnapshotORM

        orm = MagicMock(spec=PortfolioSnapshotORM)
        orm.id = "test-id"
        orm.timestamp = datetime.now(UTC)
        orm.balance = 100000.0
        orm.available_cash = 80000.0
        orm.total_exposure = 20000.0
        orm.portfolio_value = 100000.0
        orm.positions = {"AAPL": 10.0}
        orm.trigger = "TRADE"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = orm
        mock_session.execute.return_value = mock_result

        repo = PortfolioSnapshotRepository(mock_session)
        snapshot = await repo.get_latest()

        assert snapshot is not None
        assert snapshot.balance == 100000.0

    def test_repr(self, mock_session):
        """Test string representation."""
        repo = PortfolioSnapshotRepository(mock_session)
        assert repr(repo) == "PortfolioSnapshotRepository()"
