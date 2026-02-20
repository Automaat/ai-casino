"""Tests for ExecuteTradeTool with database persistence and notifications."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.daemon.config.base import TradingMode
from src.data.broker import OrderStatus
from src.metrics.tracker import TradeRecord
from src.strategies.signal import Signal
from src.v1.coordinator.tools.execute_trade import (
    DEFAULT_STOP_LOSS_PCT,
    ExecuteTradeServices,
    ExecuteTradeTool,
)


def _make_daemon_config(trading_mode=TradingMode.PAPER):
    config = Mock()
    config.trading_mode = trading_mode
    config.coordinator.min_confidence_to_trade = 0.6
    config.coordinator.confirmation_mode = "auto"
    return config


def _make_order_status(symbol="AAPL", side="buy", qty=10, filled_avg_price=150.0):
    return OrderStatus(
        order_id="order-123",
        symbol=symbol,
        qty=qty,
        filled_qty=qty,
        side=side,
        status="filled",
        submitted_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
        filled_at=datetime(2024, 6, 1, 12, 0, 1, tzinfo=UTC),
        filled_avg_price=filled_avg_price,
    )


def _make_db_engine():
    engine = AsyncMock()
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    # Return None from scalar_one_or_none so duplicate check finds no existing position
    execute_result = Mock()
    execute_result.scalar_one_or_none = Mock(return_value=None)
    session.execute = AsyncMock(return_value=execute_result)
    engine.session = Mock(return_value=session)
    return engine


@pytest.fixture
def broker():
    mock = Mock()
    mock.submit_order = Mock(return_value=_make_order_status())
    return mock


@pytest.fixture
def daemon_config():
    return _make_daemon_config()


@pytest.fixture
def database_engine():
    return _make_db_engine()


@pytest.fixture
def notification_service():
    return AsyncMock()


@pytest.fixture
def tool(broker, daemon_config):
    return ExecuteTradeTool(broker, daemon_config)


@pytest.fixture
def tool_with_db(broker, daemon_config, database_engine):
    return ExecuteTradeTool(broker, daemon_config, ExecuteTradeServices(database_engine=database_engine))


@pytest.fixture
def tool_full(broker, daemon_config, database_engine, notification_service):
    return ExecuteTradeTool(
        broker,
        daemon_config,
        ExecuteTradeServices(database_engine=database_engine, notification_service=notification_service),
    )


_TRADE_KWARGS = {
    "symbol": "AAPL",
    "action": "BUY",
    "quantity": 10,
    "confidence": 0.8,
    "rationale": "Strong momentum breakout detected",
}


@pytest.mark.asyncio
async def test_aexecute_persists_trade(tool_with_db, broker):
    result = await tool_with_db.aexecute(**_TRADE_KWARGS)

    assert "Trade Executed" in result
    assert "order-123" in result
    broker.submit_order.assert_called_once()


@pytest.mark.asyncio
async def test_aexecute_no_db_still_works(tool, broker):
    result = await tool.aexecute(**_TRADE_KWARGS)

    assert "Trade Executed" in result
    broker.submit_order.assert_called_once()


@pytest.mark.asyncio
async def test_persist_trade_failure_does_not_block(tool_with_db, broker):
    """DB error during trade persistence should not block the trade (non-critical path)."""
    with patch("src.database.repositories.trade.TradeRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.get_entry_trade = AsyncMock(return_value=None)
        mock_repo.create = AsyncMock(side_effect=RuntimeError("DB down"))
        mock_repo_cls.return_value = mock_repo

        result = await tool_with_db.aexecute(**_TRADE_KWARGS)

    assert "Trade Executed" in result


@pytest.mark.asyncio
async def test_persist_trade_creates_record(tool_with_db, database_engine):
    with patch("src.database.repositories.trade.TradeRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        await tool_with_db._persist_trade(
            _make_order_status(symbol="TSLA", side="buy", filled_avg_price=200.0),
            confidence=0.85,
            stop_loss_price=190.0,
        )

        mock_repo.create.assert_called_once()
        trade = mock_repo.create.call_args[0][0]
        assert trade.symbol == "TSLA"
        assert trade.action.value == "BUY"
        assert trade.entry_price == 200.0
        assert trade.shares == 10
        assert trade.stop_loss_price == 190.0
        assert trade.confidence == 0.85
        assert trade.risk_level == "LOW"
        assert trade.strategy_name == "coordinator"
        assert trade.broker_order_id == "order-123"
        assert trade.is_paper_trade is True
        assert trade.status == "OPEN"


def test_derive_risk_level():
    assert ExecuteTradeTool._derive_risk_level(0.90) == "LOW"
    assert ExecuteTradeTool._derive_risk_level(0.75) == "LOW"
    assert ExecuteTradeTool._derive_risk_level(0.60) == "MEDIUM"
    assert ExecuteTradeTool._derive_risk_level(0.50) == "MEDIUM"
    assert ExecuteTradeTool._derive_risk_level(0.49) == "HIGH"
    assert ExecuteTradeTool._derive_risk_level(0.10) == "HIGH"


def test_default_stop_loss_buy():
    result = ExecuteTradeTool._default_stop_loss(100.0, "buy")
    assert result == pytest.approx(100.0 * (1 - DEFAULT_STOP_LOSS_PCT))


def test_default_stop_loss_sell():
    result = ExecuteTradeTool._default_stop_loss(100.0, "sell")
    assert result == pytest.approx(100.0 * (1 + DEFAULT_STOP_LOSS_PCT))


@pytest.mark.asyncio
async def test_default_stop_loss_used_when_none(tool_with_db, database_engine):
    with patch("src.database.repositories.trade.TradeRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        await tool_with_db._persist_trade(
            _make_order_status(filled_avg_price=100.0),
            confidence=0.7,
            stop_loss_price=None,
        )

        trade = mock_repo.create.call_args[0][0]
        assert trade.stop_loss_price == pytest.approx(95.0)


@pytest.mark.asyncio
async def test_aexecute_sends_notification(tool_full, notification_service, broker):
    result = await tool_full.aexecute(**_TRADE_KWARGS)

    assert "Trade Executed" in result
    notification_service.notify.assert_called_once()

    call_args = notification_service.notify.call_args
    message = call_args[0][0]
    assert message.title == "BUY AAPL x10"
    assert message.metadata["action"] == "BUY"


@pytest.mark.asyncio
async def test_notification_failure_does_not_block(broker, daemon_config, database_engine):
    bad_notifier = AsyncMock()
    bad_notifier.notify = AsyncMock(side_effect=RuntimeError("Telegram down"))

    tool = ExecuteTradeTool(
        broker,
        daemon_config,
        ExecuteTradeServices(database_engine=database_engine, notification_service=bad_notifier),
    )

    result = await tool.aexecute(**_TRADE_KWARGS)
    assert "Trade Executed" in result


@pytest.mark.asyncio
async def test_no_notification_without_service(tool_with_db, broker):
    result = await tool_with_db.aexecute(**_TRADE_KWARGS)
    assert "Trade Executed" in result


def _make_open_buy_trade(symbol: str = "AAPL") -> TradeRecord:
    return TradeRecord(
        timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
        symbol=symbol,
        action=Signal.BUY,
        entry_price=150.0,
        exit_price=None,
        shares=10,
        stop_loss_price=142.5,
        confidence=0.8,
        risk_level="LOW",
        status="OPEN",
        pnl=None,
        pnl_percent=None,
        strategy_name="coordinator",
    )


@pytest.mark.asyncio
async def test_duplicate_buy_blocked_when_open_position_exists(broker, daemon_config, database_engine):
    """Duplicate BUY guard: existing open position should skip order and not call submit_order."""
    existing_trade = _make_open_buy_trade("AAPL")

    with patch("src.database.repositories.trade.TradeRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.get_entry_trade = AsyncMock(return_value=existing_trade)
        mock_repo_cls.return_value = mock_repo

        tool = ExecuteTradeTool(broker, daemon_config, ExecuteTradeServices(database_engine=database_engine))
        result = await tool.aexecute(**_TRADE_KWARGS)

    assert "Skipped" in result
    assert "AAPL" in result
    broker.submit_order.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_buy_allowed_when_no_open_position(broker, daemon_config, database_engine):
    """Duplicate BUY guard: no existing position should allow order through."""
    with patch("src.database.repositories.trade.TradeRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.get_entry_trade = AsyncMock(return_value=None)
        mock_repo_cls.return_value = mock_repo

        tool = ExecuteTradeTool(broker, daemon_config, ExecuteTradeServices(database_engine=database_engine))
        result = await tool.aexecute(**_TRADE_KWARGS)

    assert "Trade Executed" in result
    broker.submit_order.assert_called_once()


@pytest.mark.asyncio
async def test_duplicate_check_fails_closed_on_db_error(broker, daemon_config, database_engine):
    """DB error during duplicate check should block BUY (fail closed), not allow it."""
    session = database_engine.session()
    session.__aenter__ = AsyncMock(side_effect=RuntimeError("DB connection lost"))

    tool = ExecuteTradeTool(broker, daemon_config, ExecuteTradeServices(database_engine=database_engine))
    result = await tool.aexecute(**_TRADE_KWARGS)

    assert "Skipped" in result
    broker.submit_order.assert_not_called()
