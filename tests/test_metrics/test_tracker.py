"""Integration tests for MetricsTracker."""

import json
from datetime import UTC, datetime
from unittest.mock import mock_open, patch

import pytest

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
from src.metrics.tracker import MetricsTracker, PerformanceMetrics, TradeRecord
from src.strategies.momentum import Signal
from src.workflows.trading import TradingWorkflowResult


@pytest.fixture
def mock_workflow_result_approved():
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
        decision=TradingDecision(
            action=Signal.BUY,
            confidence=0.85,
            reasoning="Strong technical and sentiment signals",
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


@pytest.fixture
def mock_workflow_result_rejected():
    """Mock rejected trading workflow result."""
    return TradingWorkflowResult(
        symbol="TSLA",
        technical=TechnicalAnalysis(
            signal=Signal.SELL,
            rsi=75.0,
            macd_hist=-0.3,
            interpretation="Bearish momentum",
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
        decision=TradingDecision(
            action=Signal.SELL,
            confidence=0.4,
            reasoning="Weak signals, high risk",
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
                reasoning="Limited position due to risk",
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
                warnings=["Insufficient cash", "High exposure"],
                constraints_met={"max_risk": False, "cash": False},
                reasoning="Risk constraints violated",
            ),
            confidence=0.4,
        ),
    )


@pytest.fixture
def tracker_with_mocked_file():
    """MetricsTracker with mocked file I/O."""
    with patch("pathlib.Path.exists", return_value=False):
        return MetricsTracker(risk_free_rate=0.02)


def test_metrics_tracker_initialization():
    """Test MetricsTracker initialization."""
    with patch("pathlib.Path.exists", return_value=False):
        tracker = MetricsTracker(risk_free_rate=0.03)

    assert tracker.risk_free_rate == 0.03
    assert len(tracker.trades) == 0


def test_metrics_tracker_initialization_with_env():
    """Test MetricsTracker uses env var for risk-free rate."""
    with patch("pathlib.Path.exists", return_value=False):
        with patch.dict("os.environ", {"RISK_FREE_RATE": "0.05"}):
            tracker = MetricsTracker()

    assert tracker.risk_free_rate == 0.05


def test_record_decision_approved(tracker_with_mocked_file, mock_workflow_result_approved):
    """Test recording an approved trading decision."""
    with patch("pathlib.Path.open", mock_open()):
        with patch("pathlib.Path.mkdir"):
            trade = tracker_with_mocked_file.record_decision(mock_workflow_result_approved)

    assert trade.symbol == "AAPL"
    assert trade.action == Signal.BUY
    assert trade.status == "OPEN"
    assert trade.entry_price == 150.0
    assert trade.shares == 100
    assert trade.stop_loss_price == 147.0
    assert trade.confidence == 0.85
    assert trade.risk_level == "LOW"
    assert len(tracker_with_mocked_file.trades) == 1


def test_record_decision_rejected(tracker_with_mocked_file, mock_workflow_result_rejected):
    """Test recording a rejected trading decision."""
    with patch("pathlib.Path.open", mock_open()):
        with patch("pathlib.Path.mkdir"):
            trade = tracker_with_mocked_file.record_decision(mock_workflow_result_rejected)

    assert trade.symbol == "TSLA"
    assert trade.action == Signal.SELL
    assert trade.status == "REJECTED"
    assert trade.shares == 0
    assert len(tracker_with_mocked_file.trades) == 1


def test_simulate_exits_stop_loss_hit_buy(tracker_with_mocked_file):
    """Test simulating exit when stop-loss is hit for BUY trade."""
    trade = TradeRecord(
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
    tracker_with_mocked_file.trades.append(trade)

    current_prices = {"AAPL": 146.0}

    with patch("pathlib.Path.open", mock_open()):
        with patch("pathlib.Path.mkdir"):
            closed_trades = tracker_with_mocked_file.simulate_exits(current_prices)

    assert len(closed_trades) == 1
    assert closed_trades[0].status == "CLOSED"
    assert closed_trades[0].exit_price == 146.0
    assert closed_trades[0].pnl == -400.0


def test_simulate_exits_stop_loss_hit_sell(tracker_with_mocked_file):
    """Test simulating exit when stop-loss is hit for SELL trade."""
    trade = TradeRecord(
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
    tracker_with_mocked_file.trades.append(trade)

    current_prices = {"TSLA": 205.0}

    with patch("pathlib.Path.open", mock_open()):
        with patch("pathlib.Path.mkdir"):
            closed_trades = tracker_with_mocked_file.simulate_exits(current_prices)

    assert len(closed_trades) == 1
    assert closed_trades[0].status == "CLOSED"
    assert closed_trades[0].exit_price == 205.0
    assert closed_trades[0].pnl == -250.0


def test_simulate_exits_stop_loss_not_hit(tracker_with_mocked_file):
    """Test simulating exits when stop-loss is not hit."""
    trade = TradeRecord(
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
    tracker_with_mocked_file.trades.append(trade)

    current_prices = {"AAPL": 155.0}

    with patch("builtins.open", mock_open()):
        closed_trades = tracker_with_mocked_file.simulate_exits(current_prices)

    assert len(closed_trades) == 0
    assert trade.status == "OPEN"


def test_simulate_exits_missing_price_data(tracker_with_mocked_file):
    """Test simulating exits when price data is missing."""
    trade = TradeRecord(
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
    tracker_with_mocked_file.trades.append(trade)

    current_prices = {}

    with patch("builtins.open", mock_open()):
        closed_trades = tracker_with_mocked_file.simulate_exits(current_prices)

    assert len(closed_trades) == 0
    assert trade.status == "OPEN"


def test_calculate_metrics_all_window(tracker_with_mocked_file):
    """Test calculating metrics for all-time window."""
    trade1 = TradeRecord(
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
    )

    trade2 = TradeRecord(
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
    )

    tracker_with_mocked_file.trades.extend([trade1, trade2])

    metrics = tracker_with_mocked_file.calculate_metrics("all")

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.window == "all"
    assert metrics.total_decisions == 2
    assert metrics.closed_trades == 2
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate == 50.0
    assert metrics.total_pnl == 500.0


def test_calculate_metrics_empty_trades(tracker_with_mocked_file):
    """Test calculating metrics with no trades."""
    metrics = tracker_with_mocked_file.calculate_metrics("all")

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_decisions == 0
    assert metrics.win_rate == 0.0
    assert metrics.total_pnl == 0.0


def test_save_report(tracker_with_mocked_file):
    """Test saving metrics report."""
    trade = TradeRecord(
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
    )
    tracker_with_mocked_file.trades.append(trade)

    mock_file = mock_open()
    with patch("pathlib.Path.open", mock_file):
        with patch("pathlib.Path.mkdir"):
            tracker_with_mocked_file.save_report("logs/test_metrics.json")

    mock_file.assert_called_once()
    written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
    report_data = json.loads(written_content)

    assert "generated_at" in report_data
    assert "risk_free_rate" in report_data
    assert "all_time" in report_data
    assert "last_30_days" in report_data
    assert "last_7_days" in report_data


def test_load_trades_from_jsonl():
    """Test loading trades from JSONL file."""
    trade_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "symbol": "AAPL",
        "action": "BUY",
        "entry_price": 150.0,
        "exit_price": 160.0,
        "shares": 100,
        "stop_loss_price": 147.0,
        "confidence": 0.8,
        "risk_level": "LOW",
        "status": "CLOSED",
        "pnl": 1000.0,
        "pnl_percent": 6.67,
    }

    jsonl_content = json.dumps(trade_data) + "\n"

    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.open", mock_open(read_data=jsonl_content)):
            tracker = MetricsTracker()

    assert len(tracker.trades) == 1
    assert tracker.trades[0].symbol == "AAPL"
    assert tracker.trades[0].pnl == 1000.0


def test_filter_trades_by_window_30d(tracker_with_mocked_file):
    """Test filtering trades by 30-day window."""
    from datetime import timedelta

    now = datetime.now(UTC)
    old_trade = TradeRecord(
        timestamp=now - timedelta(days=45),
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

    recent_trade = TradeRecord(
        timestamp=now - timedelta(days=15),
        symbol="TSLA",
        action=Signal.SELL,
        entry_price=200.0,
        exit_price=190.0,
        shares=50,
        stop_loss_price=205.0,
        confidence=0.6,
        risk_level="MEDIUM",
        status="CLOSED",
        pnl=500.0,
        pnl_percent=5.0,
    )

    tracker_with_mocked_file.trades.extend([old_trade, recent_trade])

    filtered = tracker_with_mocked_file._filter_trades_by_window("30d")

    assert len(filtered) == 1
    assert filtered[0].symbol == "TSLA"


def test_repr(tracker_with_mocked_file):
    """Test MetricsTracker string representation."""
    repr_str = repr(tracker_with_mocked_file)

    assert "MetricsTracker" in repr_str
    assert "trades=0" in repr_str
    assert "risk_free_rate=0.02" in repr_str
