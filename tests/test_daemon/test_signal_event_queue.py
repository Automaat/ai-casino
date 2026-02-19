"""Tests for SignalEvent, next_regular_open(), and orchestrator signal emission."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from src.daemon.analysis_orchestrator import AnalysisOrchestrator
from src.daemon.config import AnalysisOrchestratorConfig
from src.daemon.events import SignalEvent
from src.daemon.factory import DaemonComponents
from src.daemon.scheduler import MarketScheduler
from src.strategies.session import TradingSession
from src.strategies.signal import Signal
from src.workflows.types import TradingDecision, TradingWorkflowResult


class TestSignalEvent:
    """Tests for SignalEvent model."""

    def test_to_prompt_text(self) -> None:
        event = SignalEvent(
            symbol="AAPL", signal="BUY", confidence=0.75, session="PRE_MARKET", reasoning="RSI oversold"
        )
        text = event.to_prompt_text()
        assert "BUY" in text
        assert "AAPL" in text
        assert "75%" in text
        assert "PRE_MARKET" in text
        assert "RSI oversold" in text

    def test_repr(self) -> None:
        event = SignalEvent(
            symbol="TSLA", signal="SELL", confidence=0.60, session="PRE_MARKET", reasoning="Breakdown"
        )
        r = repr(event)
        assert "TSLA" in r
        assert "SELL" in r
        assert "0.60" in r

    def test_default_event_type(self) -> None:
        event = SignalEvent(
            symbol="MSFT", signal="BUY", confidence=0.8, session="PRE_MARKET", reasoning="Momentum"
        )
        assert event.event_type == "signal"
        assert event.source == "analysis_orchestrator"

    def test_event_id_is_uuid(self) -> None:
        e1 = SignalEvent(symbol="AAPL", signal="BUY", confidence=0.7, session="PRE_MARKET", reasoning="r")
        e2 = SignalEvent(symbol="AAPL", signal="BUY", confidence=0.7, session="PRE_MARKET", reasoning="r")
        assert e1.event_id != e2.event_id


class TestNextRegularOpen:
    """Tests for MarketScheduler.next_regular_open()."""

    def test_returns_future_weekday(self) -> None:
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")
        # Wednesday at 10:00 AM — past open, so should return Thursday 9:30 AM
        mock_time = datetime(2024, 1, 17, 10, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            result = scheduler.next_regular_open()
        assert result > mock_time
        assert result.weekday() < 5
        assert result.hour == 9
        assert result.minute == 30

    def test_before_open_same_day(self) -> None:
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")
        # Wednesday at 7:00 AM — before today's open, return today 9:30
        mock_time = datetime(2024, 1, 17, 7, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            result = scheduler.next_regular_open()
        assert result.date() == mock_time.date()
        assert result.hour == 9
        assert result.minute == 30

    def test_friday_after_close_returns_monday(self) -> None:
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")
        # Friday at 17:00 — next open is Monday
        mock_time = datetime(2024, 1, 19, 17, 0, 0, tzinfo=tz)  # Friday
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            result = scheduler.next_regular_open()
        assert result.weekday() == 0  # Monday
        assert result.hour == 9
        assert result.minute == 30


class TestOrchestratorSignalEmission:
    """Tests for orchestrator signal event emission."""

    @pytest.fixture
    def mock_components(self) -> DaemonComponents:
        scheduler = Mock(spec=MarketScheduler)
        scheduler.get_trading_session.return_value = TradingSession.PRE_MARKET
        scheduler.next_regular_open.return_value = datetime(
            2024, 1, 18, 9, 30, 0, tzinfo=ZoneInfo("America/New_York")
        )

        components = Mock(spec=DaemonComponents)
        components.workflow = Mock()
        components.workflow.analyze = AsyncMock()
        components.state = Mock()
        components.state.record_analysis = AsyncMock()
        components.state.record_error = AsyncMock()
        components.state.get_all_positions = AsyncMock(return_value=[])
        components.state.get_active_constraints = AsyncMock(return_value=[])
        components.scheduler = scheduler
        components.broker = None
        components.position_manager = None
        components.event_bus = None
        components.historical_cache = None
        components.notification_service = None
        components.economic_calendar_watcher = None
        components.options_flow_watcher = None
        components.social_sentiment_watcher = None
        components.container = None
        components.broker_manager = Mock()
        components.broker_manager.config = Mock()
        return components

    def _make_result(
        self,
        action: Signal,
        confidence: float,
        session: TradingSession,
    ) -> TradingWorkflowResult:
        result = Mock(spec=TradingWorkflowResult)
        result.symbol = "AAPL"
        result.decision = Mock(spec=TradingDecision)
        result.decision.action = action
        result.decision.confidence = confidence
        result.decision.reasoning = ["Strong momentum"]
        result.order = None
        result.trading_session = session
        result.technical = None
        result.sentiment = None
        result.news = None
        result.risk = Mock(current_price=150.0)
        result.regime = None
        result.strategy_used = "momentum"
        return result

    @pytest.mark.asyncio
    async def test_pre_market_buy_emits_signal(self, mock_components: DaemonComponents) -> None:
        config = AnalysisOrchestratorConfig()
        orchestrator = AnalysisOrchestrator(config=config, components=mock_components)
        queue = AsyncMock()
        orchestrator.market_event_queue = queue

        result = self._make_result(Signal.BUY, 0.8, TradingSession.PRE_MARKET)
        handle = orchestrator._handle_notifications
        await handle(result)

        queue.enqueue.assert_called_once()
        call_args = queue.enqueue.call_args
        signal_event = call_args[0][0]
        assert isinstance(signal_event, SignalEvent)
        assert signal_event.signal == "BUY"
        assert signal_event.symbol == "AAPL"
        assert call_args[1]["process_after"] is not None

    @pytest.mark.asyncio
    async def test_regular_session_does_not_emit(self, mock_components: DaemonComponents) -> None:
        config = AnalysisOrchestratorConfig()
        orchestrator = AnalysisOrchestrator(config=config, components=mock_components)
        queue = AsyncMock()
        orchestrator.market_event_queue = queue

        result = self._make_result(Signal.BUY, 0.8, TradingSession.REGULAR)
        handle = orchestrator._handle_notifications
        await handle(result)

        queue.enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_confidence_does_not_emit(self, mock_components: DaemonComponents) -> None:
        config = AnalysisOrchestratorConfig()
        orchestrator = AnalysisOrchestrator(config=config, components=mock_components)
        queue = AsyncMock()
        orchestrator.market_event_queue = queue

        result = self._make_result(Signal.BUY, 0.3, TradingSession.PRE_MARKET)
        handle = orchestrator._handle_notifications
        await handle(result)

        queue.enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_hold_does_not_emit(self, mock_components: DaemonComponents) -> None:
        config = AnalysisOrchestratorConfig()
        orchestrator = AnalysisOrchestrator(config=config, components=mock_components)
        queue = AsyncMock()
        orchestrator.market_event_queue = queue

        result = self._make_result(Signal.HOLD, 0.9, TradingSession.PRE_MARKET)
        handle = orchestrator._handle_notifications
        await handle(result)

        queue.enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_queue_wired_skips_emit(self, mock_components: DaemonComponents) -> None:
        """No exception raised when market_event_queue is None."""
        config = AnalysisOrchestratorConfig()
        orchestrator = AnalysisOrchestrator(config=config, components=mock_components)
        # market_event_queue stays None

        result = self._make_result(Signal.BUY, 0.8, TradingSession.PRE_MARKET)
        handle = orchestrator._handle_notifications
        await handle(result)
        # No assertion needed — just verifying no exception

    @pytest.mark.asyncio
    async def test_sell_signal_emits(self, mock_components: DaemonComponents) -> None:
        config = AnalysisOrchestratorConfig()
        orchestrator = AnalysisOrchestrator(config=config, components=mock_components)
        queue = AsyncMock()
        orchestrator.market_event_queue = queue

        result = self._make_result(Signal.SELL, 0.75, TradingSession.PRE_MARKET)
        handle = orchestrator._handle_notifications
        await handle(result)

        queue.enqueue.assert_called_once()
        signal_event = queue.enqueue.call_args[0][0]
        assert signal_event.signal == "SELL"
