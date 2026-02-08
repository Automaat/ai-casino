"""Tests for DatabaseMetricsTracker."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.bearish_researcher import BearishResearchAnalysis
from src.agents.bullish_researcher import BullishResearchAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.risk import (
    AccountInfo,
    PositionSizeCalculation,
    RiskAssessment,
    RiskValidation,
    StopLossCalculation,
)
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trader import TradingDecision
from src.metrics.tracker import DatabaseMetricsTracker, PerformanceMetrics, TradeRecord
from src.strategies.signal import Signal
from src.workflows.types import TradingWorkflowResult


@pytest.fixture
def mock_trade_repo():
    """Mock trade repository for testing."""
    mock = MagicMock()

    trade_record = TradeRecord(
        id="test-uuid-1234",
        timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        symbol="AAPL",
        action=Signal.BUY,
        entry_price=150.0,
        exit_price=None,
        shares=10,
        stop_loss_price=145.0,
        confidence=0.75,
        risk_level="MEDIUM",
        status="OPEN",
        pnl=None,
        pnl_percent=None,
        strategy_name="momentum",
    )

    mock.create = AsyncMock(return_value=trade_record)
    mock.get_by_id = AsyncMock(return_value=trade_record)
    mock.get_open_trades = AsyncMock(return_value=[trade_record])
    mock.get_by_window = AsyncMock(return_value=[trade_record])
    mock.get_by_symbol = AsyncMock(return_value=[trade_record])
    mock.get_all = AsyncMock(return_value=[trade_record])
    mock.update = AsyncMock(return_value=trade_record)

    return mock


@pytest.fixture
def db_tracker(mock_trade_repo):
    """DatabaseMetricsTracker with mocked repository."""
    return DatabaseMetricsTracker(mock_trade_repo, risk_free_rate=0.02)


@pytest.fixture
def mock_workflow_result():
    """Mock approved trading workflow result."""
    return TradingWorkflowResult(
        symbol="AAPL",
        technical=TechnicalAnalysis(
            signal=Signal.BUY,
            rsi=35.0,
            macd_hist=0.5,
            interpretation="Bullish momentum",
            confidence=0.8,
        ),
        sentiment=SentimentAnalysis(
            overall_sentiment="POSITIVE",
            sentiment_score=0.6,
            positive_ratio=0.7,
            negative_ratio=0.1,
            neutral_ratio=0.2,
            article_count=5,
            summary="Strong positive sentiment",
        ),
        news=NewsAnalysis(
            key_themes=["earnings", "growth"],
            impact_assessment="POSITIVE",
            recommendation="BUY",
        ),
        fundamental=FundamentalAnalysis(
            valuation="FAIRLY_VALUED",
            pe_ratio=28.5,
            eps=6.15,
            revenue_growth_yoy=0.062,
            earnings_growth_yoy=0.102,
            debt_to_equity=2.05,
            current_ratio=0.94,
            interpretation="Solid fundamentals",
            confidence=0.75,
        ),
        bullish=BullishResearchAnalysis(
            thesis="Strong momentum supports upside potential.",
            key_strengths=["Bullish technical signals", "Positive sentiment"],
            target_upside=15.0,
            confidence=0.8,
        ),
        bearish=BearishResearchAnalysis(
            thesis="Limited downside risk given strong fundamentals.",
            key_weaknesses=["Some market uncertainty"],
            target_downside=5.0,
            confidence=0.3,
        ),
        decision=TradingDecision(
            action=Signal.BUY,
            confidence=0.85,
            reasoning=["Strong technical and sentiment signals"],
            risk_level="LOW",
        ),
        risk=RiskAssessment(
            symbol="AAPL",
            action=Signal.BUY,
            current_price=150.0,
            account_info=AccountInfo(
                balance=100000.0,
                available_cash=80000.0,
                positions={},
                total_exposure=0.0,
            ),
            position_sizing=PositionSizeCalculation(
                recommended_shares=100,
                position_value=15000.0,
                risk_amount=300.0,
                risk_percent=2.0,
                reasoning="Standard position sizing",
            ),
            stop_loss=StopLossCalculation(
                stop_loss_price=147.0,
                stop_loss_percent=2.0,
                risk_per_share=3.0,
                max_loss_amount=300.0,
                methodology="ATR",
            ),
            validation=RiskValidation(
                approved=True,
                risk_score=0.2,
                risk_level="LOW",
                warnings=[],
                constraints_met={"max_risk": True, "cash": True},
                reasoning="All constraints met",
            ),
            confidence=0.85,
        ),
    )


class TestDatabaseMetricsTrackerInit:
    """Tests for DatabaseMetricsTracker initialization."""

    def test_initialization(self, mock_trade_repo):
        """Test DatabaseMetricsTracker initialization."""
        tracker = DatabaseMetricsTracker(mock_trade_repo, risk_free_rate=0.03)

        assert tracker.risk_free_rate == 0.03
        assert tracker._repo == mock_trade_repo
        assert tracker._trades_cache is None

    def test_initialization_default_risk_rate(self, mock_trade_repo):
        """Test DatabaseMetricsTracker uses default risk-free rate."""
        tracker = DatabaseMetricsTracker(mock_trade_repo)

        assert tracker.risk_free_rate == 0.02

    def test_repr(self, db_tracker):
        """Test DatabaseMetricsTracker string representation."""
        repr_str = repr(db_tracker)

        assert "DatabaseMetricsTracker" in repr_str
        assert "risk_free_rate=0.02" in repr_str


class TestRecordDecisionAsync:
    """Tests for record_decision_async method."""

    async def test_record_decision_async_approved(self, db_tracker, mock_workflow_result, mock_trade_repo):
        """Test recording an approved trading decision asynchronously."""
        trade = await db_tracker.record_decision_async(mock_workflow_result, strategy_name="momentum")

        assert trade.symbol == "AAPL"
        assert trade.action == Signal.BUY
        mock_trade_repo.create.assert_called_once()

    async def test_record_decision_async_invalidates_cache(self, db_tracker, mock_workflow_result):
        """Test that recording a decision invalidates the cache."""
        db_tracker._trades_cache = [MagicMock()]

        await db_tracker.record_decision_async(mock_workflow_result)

        assert db_tracker._trades_cache is None

    async def test_record_decision_async_rejected(self, db_tracker, mock_trade_repo):
        """Test recording a rejected trading decision."""
        result = TradingWorkflowResult(
            symbol="TSLA",
            technical=TechnicalAnalysis(
                signal=Signal.SELL,
                rsi=75.0,
                macd_hist=-0.3,
                interpretation="Bearish",
                confidence=0.5,
            ),
            sentiment=SentimentAnalysis(
                overall_sentiment="NEGATIVE",
                sentiment_score=-0.4,
                positive_ratio=0.2,
                negative_ratio=0.6,
                neutral_ratio=0.2,
                article_count=3,
                summary="Negative sentiment",
            ),
            news=NewsAnalysis(
                key_themes=["volatility"],
                impact_assessment="NEGATIVE",
                recommendation="SELL",
            ),
            fundamental=FundamentalAnalysis(
                valuation="OVERVALUED",
                pe_ratio=45.0,
                eps=3.2,
                revenue_growth_yoy=-0.05,
                earnings_growth_yoy=-0.12,
                debt_to_equity=3.5,
                current_ratio=0.8,
                interpretation="Weak fundamentals",
                confidence=0.6,
            ),
            bullish=BullishResearchAnalysis(
                thesis="Limited upside.",
                key_strengths=[],
                target_upside=None,
                confidence=0.3,
            ),
            bearish=BearishResearchAnalysis(
                thesis="Significant downside risk.",
                key_weaknesses=["Weak fundamentals"],
                target_downside=25.0,
                confidence=0.8,
            ),
            decision=TradingDecision(
                action=Signal.SELL,
                confidence=0.4,
                reasoning=["Weak signals"],
                risk_level="HIGH",
            ),
            risk=RiskAssessment(
                symbol="TSLA",
                action=Signal.SELL,
                current_price=200.0,
                account_info=AccountInfo(
                    balance=100000.0,
                    available_cash=10000.0,
                    positions={},
                    total_exposure=90000.0,
                ),
                position_sizing=PositionSizeCalculation(
                    recommended_shares=50,
                    position_value=10000.0,
                    risk_amount=200.0,
                    risk_percent=2.0,
                    reasoning="Limited position",
                ),
                stop_loss=StopLossCalculation(
                    stop_loss_price=204.0,
                    stop_loss_percent=2.0,
                    risk_per_share=4.0,
                    max_loss_amount=200.0,
                    methodology="Fixed",
                ),
                validation=RiskValidation(
                    approved=False,
                    risk_score=0.8,
                    risk_level="HIGH",
                    warnings=["Insufficient cash"],
                    constraints_met={"max_risk": False, "cash": False},
                    reasoning="Risk constraints violated",
                ),
                confidence=0.4,
            ),
        )

        await db_tracker.record_decision_async(result)

        call_args = mock_trade_repo.create.call_args[0][0]
        assert call_args.status == "REJECTED"
        assert call_args.shares == 0


class TestSimulateExitsAsync:
    """Tests for simulate_exits_async method."""

    async def test_simulate_exits_async_stop_loss_hit(self, db_tracker, mock_trade_repo):
        """Test simulating exit when stop-loss is hit."""
        open_trade = TradeRecord(
            id="trade-123",
            timestamp=datetime.now(UTC),
            symbol="AAPL",
            action=Signal.BUY,
            entry_price=150.0,
            exit_price=None,
            shares=100,
            stop_loss_price=147.0,
            confidence=0.8,
            risk_level="LOW",
            status="OPEN",
            pnl=None,
            pnl_percent=None,
        )
        mock_trade_repo.get_open_trades = AsyncMock(return_value=[open_trade])

        current_prices = {"AAPL": 146.0}
        closed_trades = await db_tracker.simulate_exits_async(current_prices)

        assert len(closed_trades) == 1
        assert closed_trades[0].status == "CLOSED"
        assert closed_trades[0].exit_price == 146.0
        mock_trade_repo.update.assert_called_once_with(
            "trade-123",
            status="CLOSED",
            exit_price=146.0,
            pnl=closed_trades[0].pnl,
            pnl_percent=closed_trades[0].pnl_percent,
            closed_at=closed_trades[0].closed_at,
        )

    async def test_simulate_exits_async_stop_loss_not_hit(self, db_tracker, mock_trade_repo):
        """Test simulating exits when stop-loss is not hit."""
        open_trade = TradeRecord(
            id="trade-123",
            timestamp=datetime.now(UTC),
            symbol="AAPL",
            action=Signal.BUY,
            entry_price=150.0,
            exit_price=None,
            shares=100,
            stop_loss_price=147.0,
            confidence=0.8,
            risk_level="LOW",
            status="OPEN",
            pnl=None,
            pnl_percent=None,
        )
        mock_trade_repo.get_open_trades = AsyncMock(return_value=[open_trade])

        current_prices = {"AAPL": 155.0}
        closed_trades = await db_tracker.simulate_exits_async(current_prices)

        assert len(closed_trades) == 0
        mock_trade_repo.update.assert_not_called()

    async def test_simulate_exits_async_missing_price(self, db_tracker, mock_trade_repo):
        """Test simulating exits when price data is missing."""
        open_trade = TradeRecord(
            id="trade-123",
            timestamp=datetime.now(UTC),
            symbol="AAPL",
            action=Signal.BUY,
            entry_price=150.0,
            exit_price=None,
            shares=100,
            stop_loss_price=147.0,
            confidence=0.8,
            risk_level="LOW",
            status="OPEN",
            pnl=None,
            pnl_percent=None,
        )
        mock_trade_repo.get_open_trades = AsyncMock(return_value=[open_trade])

        current_prices = {}
        closed_trades = await db_tracker.simulate_exits_async(current_prices)

        assert len(closed_trades) == 0

    async def test_simulate_exits_async_sell_stop_loss(self, db_tracker, mock_trade_repo):
        """Test simulating exit for SELL trade when stop-loss is hit."""
        open_trade = TradeRecord(
            id="trade-456",
            timestamp=datetime.now(UTC),
            symbol="TSLA",
            action=Signal.SELL,
            entry_price=200.0,
            exit_price=None,
            shares=50,
            stop_loss_price=204.0,
            confidence=0.6,
            risk_level="MEDIUM",
            status="OPEN",
            pnl=None,
            pnl_percent=None,
        )
        mock_trade_repo.get_open_trades = AsyncMock(return_value=[open_trade])

        current_prices = {"TSLA": 205.0}
        closed_trades = await db_tracker.simulate_exits_async(current_prices)

        assert len(closed_trades) == 1
        assert closed_trades[0].status == "CLOSED"
        assert closed_trades[0].pnl == -250.0


class TestCalculateMetricsAsync:
    """Tests for calculate_metrics_async method."""

    async def test_calculate_metrics_async_with_trades(self, db_tracker, mock_trade_repo):
        """Test calculating metrics with closed trades."""
        closed_trades = [
            TradeRecord(
                id="t1",
                timestamp=datetime.now(UTC),
                symbol="AAPL",
                action=Signal.BUY,
                entry_price=100.0,
                exit_price=110.0,
                shares=100,
                stop_loss_price=95.0,
                confidence=0.8,
                risk_level="LOW",
                status="CLOSED",
                pnl=1000.0,
                pnl_percent=10.0,
            ),
            TradeRecord(
                id="t2",
                timestamp=datetime.now(UTC),
                symbol="TSLA",
                action=Signal.SELL,
                entry_price=200.0,
                exit_price=210.0,
                shares=50,
                stop_loss_price=205.0,
                confidence=0.6,
                risk_level="MEDIUM",
                status="CLOSED",
                pnl=-500.0,
                pnl_percent=-5.0,
            ),
        ]
        mock_trade_repo.get_by_window = AsyncMock(return_value=closed_trades)

        metrics = await db_tracker.calculate_metrics_async("all")

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.window == "all"
        assert metrics.total_decisions == 2
        assert metrics.closed_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.total_pnl == 500.0

    async def test_calculate_metrics_async_empty(self, db_tracker, mock_trade_repo):
        """Test calculating metrics with no trades."""
        mock_trade_repo.get_by_window = AsyncMock(return_value=[])

        metrics = await db_tracker.calculate_metrics_async("all")

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_decisions == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == 0.0

    async def test_calculate_metrics_async_30d_window(self, db_tracker, mock_trade_repo):
        """Test calculating metrics for 30-day window."""
        recent_trade = TradeRecord(
            id="t1",
            timestamp=datetime.now(UTC) - timedelta(days=15),
            symbol="AAPL",
            action=Signal.BUY,
            entry_price=100.0,
            exit_price=110.0,
            shares=100,
            stop_loss_price=95.0,
            confidence=0.8,
            risk_level="LOW",
            status="CLOSED",
            pnl=1000.0,
            pnl_percent=10.0,
        )
        mock_trade_repo.get_by_window = AsyncMock(return_value=[recent_trade])

        metrics = await db_tracker.calculate_metrics_async("30d")

        assert metrics.window == "30d"
        assert metrics.total_decisions == 1
        mock_trade_repo.get_by_window.assert_called_with("30d")


class TestSyncWrappers:
    """Tests for synchronous wrapper methods."""

    def test_record_decision_sync_wrapper(self, db_tracker, mock_workflow_result, mock_trade_repo):
        """Test sync wrapper for record_decision."""
        trade = db_tracker.record_decision(mock_workflow_result, strategy_name="momentum")

        assert trade.symbol == "AAPL"
        mock_trade_repo.create.assert_called_once()

    def test_simulate_exits_sync_wrapper(self, db_tracker, mock_trade_repo):
        """Test sync wrapper for simulate_exits."""
        mock_trade_repo.get_open_trades = AsyncMock(return_value=[])

        closed = db_tracker.simulate_exits({"AAPL": 150.0})

        assert closed == []
        mock_trade_repo.get_open_trades.assert_called_once()

    def test_calculate_metrics_sync_wrapper(self, db_tracker, mock_trade_repo):
        """Test sync wrapper for calculate_metrics."""
        mock_trade_repo.get_by_window = AsyncMock(return_value=[])

        metrics = db_tracker.calculate_metrics("all")

        assert isinstance(metrics, PerformanceMetrics)
        mock_trade_repo.get_by_window.assert_called_with("all")


class TestCacheManagement:
    """Tests for cache management."""

    async def test_get_trades_caches_result(self, db_tracker, mock_trade_repo):
        """Test that _get_trades caches the result."""
        trades = await db_tracker._get_trades()
        trades_again = await db_tracker._get_trades()

        assert trades == trades_again
        mock_trade_repo.get_all.assert_called_once()

    def test_invalidate_cache(self, db_tracker):
        """Test cache invalidation."""
        db_tracker._trades_cache = [MagicMock()]

        db_tracker._invalidate_cache()

        assert db_tracker._trades_cache is None
