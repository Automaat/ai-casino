"""Tests for TradingService."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.daemon.config.base import TradingMode
from src.data.broker import OrderStatus
from src.strategies.signal import Signal
from src.v1.trades.models import TradeAction, TradeRejectionReason, TradeRequest
from src.v1.trades.service import TradingService


def _make_daemon_config(
    trading_mode: TradingMode = TradingMode.PAPER,
    min_confidence: float = 0.6,
) -> Mock:
    config = Mock()
    config.trading_mode = trading_mode
    config.coordinator.min_confidence_to_trade = min_confidence
    config.coordinator.confirmation_mode = "auto"
    return config


def _make_order_status(
    symbol: str = "AAPL",
    side: str = "buy",
    qty: int = 10,
    filled_avg_price: float = 150.0,
) -> OrderStatus:
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


def _make_db_engine() -> AsyncMock:
    engine = AsyncMock()
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    execute_result = Mock()
    execute_result.scalar_one_or_none = Mock(return_value=None)
    session.execute = AsyncMock(return_value=execute_result)
    engine.session = Mock(return_value=session)
    return engine


_REQUEST = TradeRequest(
    symbol="AAPL",
    action=TradeAction.BUY,
    quantity=10,
    confidence=0.8,
    rationale="Strong momentum breakout detected",
)


@pytest.fixture
def broker() -> Mock:
    mock = Mock()
    mock.submit_order = Mock(return_value=_make_order_status())
    return mock


@pytest.fixture
def daemon_config() -> Mock:
    return _make_daemon_config()


@pytest.fixture
def service(broker: Mock, daemon_config: Mock) -> TradingService:
    return TradingService(broker=broker, daemon_config=daemon_config)


@pytest.fixture
def service_full(broker: Mock, daemon_config: Mock) -> TradingService:
    return TradingService(
        broker=broker,
        daemon_config=daemon_config,
        database_engine=_make_db_engine(),
        notification_service=AsyncMock(),
    )


class TestHappyPath:
    """Tests for successful trade execution."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_execute_returns_success(self, service: TradingService, broker: Mock) -> None:
        result = await service.execute(_REQUEST)

        assert result.executed is True
        assert result.order_id == "order-123"
        assert result.symbol == "AAPL"
        assert result.action == TradeAction.BUY
        broker.submit_order.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_execute_persists_and_notifies(self, service_full: TradingService) -> None:
        with patch("src.database.repositories.trade.TradeRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.get_entry_trade = AsyncMock(return_value=None)
            mock_repo_cls.return_value = mock_repo

            result = await service_full.execute(_REQUEST)

        assert result.executed is True
        mock_repo.create.assert_called_once()
        service_full._notification_service.notify.assert_called_once()
        notification_msg = service_full._notification_service.notify.call_args[0][0]
        assert notification_msg.title == "💰 BUY AAPL x10"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_sell_notification_title(self, broker: Mock, daemon_config: Mock) -> None:
        broker.submit_order = Mock(return_value=_make_order_status(side="sell"))
        notification_service = AsyncMock()
        service = TradingService(
            broker=broker,
            daemon_config=daemon_config,
            notification_service=notification_service,
        )
        sell_request = TradeRequest(
            symbol="AAPL",
            action=TradeAction.SELL,
            quantity=10,
            confidence=0.8,
            rationale="Taking profits",
        )

        result = await service.execute(sell_request)

        assert result.executed is True
        notification_service.notify.assert_called_once()
        notification_msg = notification_service.notify.call_args[0][0]
        assert notification_msg.title == "🔴 SELL AAPL x10"


class TestThresholdRejection:
    """Tests for confidence threshold check."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_below_threshold_rejected(self, broker: Mock) -> None:
        config = _make_daemon_config(min_confidence=0.9)
        service = TradingService(broker=broker, daemon_config=config)

        result = await service.execute(_REQUEST)

        assert result.executed is False
        assert result.rejection is not None
        assert result.rejection.reason == TradeRejectionReason.BELOW_THRESHOLD
        broker.submit_order.assert_not_called()


class TestDuplicatePosition:
    """Tests for duplicate BUY guard."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_duplicate_buy_blocked(self, broker: Mock, daemon_config: Mock) -> None:
        from src.metrics.tracker import TradeRecord

        existing = TradeRecord(
            timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
            symbol="AAPL",
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

        db_engine = _make_db_engine()
        service = TradingService(broker=broker, daemon_config=daemon_config, database_engine=db_engine)

        with patch("src.database.repositories.trade.TradeRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.get_entry_trade = AsyncMock(return_value=existing)
            mock_repo_cls.return_value = mock_repo

            result = await service.execute(_REQUEST)

        assert result.executed is False
        assert result.rejection is not None
        assert result.rejection.reason == TradeRejectionReason.DUPLICATE_POSITION
        broker.submit_order.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_sell_skips_duplicate_check(self, broker: Mock, daemon_config: Mock) -> None:
        """SELL orders should not check for duplicate positions."""
        sell_request = TradeRequest(
            symbol="AAPL",
            action=TradeAction.SELL,
            quantity=10,
            confidence=0.8,
            rationale="Taking profits on position",
        )
        db_engine = _make_db_engine()
        service = TradingService(broker=broker, daemon_config=daemon_config, database_engine=db_engine)

        result = await service.execute(sell_request)

        assert result.executed is True
        broker.submit_order.assert_called_once()


class TestConfirmation:
    """Tests for manual confirmation."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_confirmation_rejected(self, broker: Mock) -> None:
        config = _make_daemon_config(trading_mode=TradingMode.LIVE)
        config.coordinator.confirmation_mode = "manual"
        handler = AsyncMock()
        handler.request_approval = AsyncMock(return_value=False)

        service = TradingService(broker=broker, daemon_config=config, confirmation_handler=handler)
        result = await service.execute(_REQUEST)

        assert result.executed is False
        assert result.rejection is not None
        assert result.rejection.reason == TradeRejectionReason.CONFIRMATION_REJECTED
        broker.submit_order.assert_not_called()


class TestBrokerError:
    """Tests for broker submission failure."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_broker_error_returns_rejection(self, daemon_config: Mock) -> None:
        broker = Mock()
        broker.submit_order = Mock(side_effect=RuntimeError("Connection refused"))
        service = TradingService(broker=broker, daemon_config=daemon_config)

        result = await service.execute(_REQUEST)

        assert result.executed is False
        assert result.rejection is not None
        assert result.rejection.reason == TradeRejectionReason.BROKER_ERROR


class TestOptionalDeps:
    """Tests that missing optional deps don't cause failures."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_no_db_still_works(self, service: TradingService) -> None:
        result = await service.execute(_REQUEST)
        assert result.executed is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_no_notification_still_works(self, service: TradingService) -> None:
        result = await service.execute(_REQUEST)
        assert result.executed is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_db_failure_non_blocking(self, broker: Mock, daemon_config: Mock) -> None:
        db_engine = _make_db_engine()
        notification = AsyncMock()
        service = TradingService(
            broker=broker,
            daemon_config=daemon_config,
            database_engine=db_engine,
            notification_service=notification,
        )

        with patch("src.database.repositories.trade.TradeRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.get_entry_trade = AsyncMock(return_value=None)
            mock_repo.create = AsyncMock(side_effect=RuntimeError("DB down"))
            mock_repo_cls.return_value = mock_repo

            result = await service.execute(_REQUEST)

        assert result.executed is True
        notification.notify.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_notification_failure_non_blocking(self, broker: Mock, daemon_config: Mock) -> None:
        bad_notifier = AsyncMock()
        bad_notifier.notify = AsyncMock(side_effect=RuntimeError("Telegram down"))
        service = TradingService(
            broker=broker, daemon_config=daemon_config, notification_service=bad_notifier
        )

        result = await service.execute(_REQUEST)
        assert result.executed is True


class TestRiskLevel:
    """Tests for risk level derivation."""

    @pytest.mark.unit
    def test_derive_risk_level(self) -> None:
        assert TradingService._derive_risk_level(0.90) == "LOW"
        assert TradingService._derive_risk_level(0.75) == "LOW"
        assert TradingService._derive_risk_level(0.60) == "MEDIUM"
        assert TradingService._derive_risk_level(0.50) == "MEDIUM"
        assert TradingService._derive_risk_level(0.49) == "HIGH"
        assert TradingService._derive_risk_level(0.10) == "HIGH"
