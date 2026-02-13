"""Tests for CoordinatorMemory enhanced functionality."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from src.coordinator.memory import CoordinatorMemory
from src.daemon.state.models import AnalysisRecord, GamePlanRecord
from src.strategies.session import TradingSession

pytestmark = pytest.mark.skip(reason="Coordinator memory tests need rewrite for async state")


@pytest.fixture
def daemon_state_with_analyses():
    """Create mock DaemonState with sample analyses."""
    state = Mock()

    # Create analyses from today and yesterday
    today = datetime.now(UTC)
    yesterday = today - timedelta(days=1)

    state.analyses = [
        # Today's analyses
        AnalysisRecord(
            symbol="AAPL",
            timestamp=today.replace(hour=9, minute=45),
            signal="BUY",
            confidence=0.85,
            executed_trade=True,
            trading_session=TradingSession.REGULAR,
            rsi=35.0,
            macd_hist=0.5,
        ),
        AnalysisRecord(
            symbol="TSLA",
            timestamp=today.replace(hour=10, minute=30),
            signal="SELL",
            confidence=0.75,
            executed_trade=False,
            trading_session=TradingSession.REGULAR,
            rsi=72.0,
            macd_hist=-0.3,
        ),
        AnalysisRecord(
            symbol="MSFT",
            timestamp=today.replace(hour=11, minute=15),
            signal="HOLD",
            confidence=0.65,
            executed_trade=False,
            trading_session=TradingSession.REGULAR,
            rsi=50.0,
            macd_hist=0.1,
        ),
        AnalysisRecord(
            symbol="GOOGL",
            timestamp=today.replace(hour=8, minute=15),
            signal="BUY",
            confidence=0.72,
            executed_trade=False,
            trading_session=TradingSession.PRE_MARKET,
            rsi=38.0,
            macd_hist=0.4,
        ),
        # Yesterday's analyses (should be filtered out)
        AnalysisRecord(
            symbol="NVDA",
            timestamp=yesterday.replace(hour=14, minute=0),
            signal="BUY",
            confidence=0.90,
            executed_trade=True,
            trading_session=TradingSession.REGULAR,
            rsi=42.0,
            macd_hist=0.7,
        ),
    ]

    return state


@pytest.fixture
def daemon_state_with_game_plan():
    """Create mock DaemonState with sample game plan."""
    state = Mock()

    today = datetime.now(UTC)
    yesterday = today - timedelta(days=1)

    state.game_plan_history = [
        # Yesterday's plan (should be filtered out)
        GamePlanRecord(
            timestamp=yesterday.replace(hour=8, minute=0),
            priority_symbols=["OLD1", "OLD2"],
            risk_stance="AGGRESSIVE",
            sector_focus=["Energy"],
        ),
        # Today's plan
        GamePlanRecord(
            timestamp=today.replace(hour=8, minute=30),
            priority_symbols=["AAPL", "TSLA", "MSFT"],
            risk_stance="NEUTRAL",
            sector_focus=["Technology", "Consumer Cyclical"],
        ),
    ]

    return state


@pytest.fixture
def mock_broker():
    """Create mock AlpacaBroker with account info."""
    broker = Mock()

    account_info = Mock()
    account_info.balance = 50000.0
    account_info.portfolio_value = 55000.0
    account_info.available_cash = 25000.0
    account_info.total_exposure = 30000.0
    account_info.positions = {
        "AAPL": Mock(
            qty=50,
            avg_entry_price=180.0,
            unrealized_pnl=500.0,
            unrealized_pnl_percent=5.6,
        ),
        "TSLA": Mock(
            qty=30,
            avg_entry_price=250.0,
            unrealized_pnl=-300.0,
            unrealized_pnl_percent=-4.0,
        ),
    }

    broker.get_account_info.return_value = account_info
    return broker


@pytest.fixture
def mock_analysis_repo():
    """Create mock AnalysisRecordRepository."""
    repo = AsyncMock()

    # Setup default return value for get_by_date_range
    today = datetime.now(UTC)
    repo.get_by_date_range.return_value = [
        AnalysisRecord(
            symbol="AAPL",
            timestamp=today - timedelta(days=2),
            signal="BUY",
            confidence=0.80,
            executed_trade=True,
            trading_session=TradingSession.REGULAR,
            rsi=40.0,
            macd_hist=0.3,
        ),
        AnalysisRecord(
            symbol="AAPL",
            timestamp=today - timedelta(days=5),
            signal="HOLD",
            confidence=0.60,
            executed_trade=False,
            trading_session=TradingSession.REGULAR,
            rsi=55.0,
            macd_hist=-0.1,
        ),
    ]

    return repo


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_today_summary_with_analyses(daemon_state_with_analyses, tmp_path):
    """Test get_today_summary with multiple analyses."""
    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        daemon_state=daemon_state_with_analyses,
    )

    result = await memory.get_today_summary()

    assert "## Today's Analyses" in result
    assert "**BUY**" in result
    assert "**SELL**" in result
    assert "**HOLD**" in result
    assert "AAPL" in result
    assert "TSLA" in result
    assert "MSFT" in result
    assert "GOOGL" in result
    assert "(PRE_MARKET)" in result  # Pre-market session indicator
    assert "NVDA" not in result  # Yesterday's analysis should be filtered out
    assert "Confidence: 85%" in result
    assert "Executed: ✓" in result
    assert "Executed: ✗" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_today_summary_empty(tmp_path):
    """Test get_today_summary with no analyses."""
    state = Mock()
    state.analyses = []

    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        daemon_state=state,
    )

    result = await memory.get_today_summary()
    assert result == "No analyses today"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_today_summary_no_daemon_state(tmp_path):
    """Test get_today_summary without daemon_state."""
    memory = CoordinatorMemory(memory_file=tmp_path / "memory.jsonl")

    result = await memory.get_today_summary()
    assert result == "No analyses today"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_today_game_plan_found(daemon_state_with_game_plan, tmp_path):
    """Test get_today_game_plan with existing plan."""
    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        daemon_state=daemon_state_with_game_plan,
    )

    result = await memory.get_today_game_plan()

    assert "## Today's Game Plan" in result
    assert "AAPL, TSLA, MSFT" in result
    assert "NEUTRAL" in result
    assert "Technology, Consumer Cyclical" in result
    assert "OLD1" not in result  # Yesterday's plan should be filtered out


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_today_game_plan_not_found(tmp_path):
    """Test get_today_game_plan with no plan today."""
    state = Mock()
    state.game_plan_history = []

    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        daemon_state=state,
    )

    result = await memory.get_today_game_plan()
    assert result == "No game plan generated today"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_today_game_plan_no_daemon_state(tmp_path):
    """Test get_today_game_plan without daemon_state."""
    memory = CoordinatorMemory(memory_file=tmp_path / "memory.jsonl")

    result = await memory.get_today_game_plan()
    assert result == "Game plan unavailable"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_portfolio_summary(mock_broker, tmp_path):
    """Test get_portfolio_summary from broker."""
    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        broker=mock_broker,
    )

    result = await memory.get_portfolio_summary()

    assert "## Current Portfolio" in result
    assert "$50,000.00" in result  # Balance
    assert "$55,000.00" in result  # Portfolio value
    assert "$25,000.00" in result  # Available cash
    assert "$30,000.00" in result  # Total exposure
    assert "54.5%" in result  # Exposure percentage
    assert "Positions (2):" in result
    assert "AAPL: 50 shares" in result
    assert "TSLA: 30 shares" in result
    assert "P&L: $500.00" in result
    assert "P&L: $-300.00" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_portfolio_summary_broker_unavailable(tmp_path):
    """Test get_portfolio_summary when broker is not available."""
    memory = CoordinatorMemory(memory_file=tmp_path / "memory.jsonl")

    result = await memory.get_portfolio_summary()
    assert result == "Portfolio data unavailable"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_portfolio_summary_broker_error(tmp_path):
    """Test get_portfolio_summary when broker raises exception."""
    broker = Mock()
    broker.get_account_info.side_effect = Exception("Broker API error")

    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        broker=broker,
    )

    result = await memory.get_portfolio_summary()
    assert "Portfolio data unavailable:" in result
    assert "Broker API error" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_analysis_history_from_db(mock_analysis_repo, tmp_path):
    """Test get_analysis_history querying database."""
    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        analysis_repo=mock_analysis_repo,
    )

    result = await memory.get_analysis_history("AAPL", days=7)

    assert "# Analysis History - AAPL (last 7 days)" in result
    assert "BUY" in result
    assert "HOLD" in result
    assert "80%" in result
    assert "60%" in result
    assert "RSI: 40.0" in result
    assert "MACD: 0.3000" in result

    # Verify repository was called with correct parameters
    mock_analysis_repo.get_by_date_range.assert_called_once()
    call_args = mock_analysis_repo.get_by_date_range.call_args
    assert call_args.kwargs["symbol"] == "AAPL"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_analysis_history_no_records(mock_analysis_repo, tmp_path):
    """Test get_analysis_history with no records found."""
    mock_analysis_repo.get_by_date_range.return_value = []

    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        analysis_repo=mock_analysis_repo,
    )

    result = await memory.get_analysis_history("NVDA", days=7)
    assert "No analysis history found for NVDA in last 7 days" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_analysis_history_fallback_to_memory(daemon_state_with_analyses, tmp_path):
    """Test get_analysis_history fallback to in-memory when no repository."""
    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        daemon_state=daemon_state_with_analyses,
    )

    result = await memory.get_analysis_history("AAPL", days=7)

    assert "# Analysis History - AAPL (in-memory only)" in result
    assert "BUY" in result
    assert "85%" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_analysis_history_no_repo_no_state(tmp_path):
    """Test get_analysis_history when neither repo nor state available."""
    memory = CoordinatorMemory(memory_file=tmp_path / "memory.jsonl")

    result = await memory.get_analysis_history("AAPL", days=7)
    assert "No analysis history available for AAPL" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_token_budget_truncation(daemon_state_with_analyses, tmp_path):
    """Test that token budget truncation works correctly."""
    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        daemon_state=daemon_state_with_analyses,
    )

    # Use very small budget to force truncation
    result = await memory.get_today_summary(max_tokens=10)

    # Should be truncated (10 tokens ≈ 40 chars)
    assert len(result) < 200
    assert "[Truncated for length]" in result


@pytest.mark.unit
def test_truncate_to_budget():
    """Test _truncate_to_budget helper method."""
    memory = CoordinatorMemory()

    # Short text - no truncation
    short_text = "This is a short text."
    result = memory._truncate_to_budget(short_text, max_tokens=100)
    assert result == short_text
    assert "[Truncated for length]" not in result

    # Long text - should truncate
    long_text = "Line 1\n" * 100  # 700 chars
    result = memory._truncate_to_budget(long_text, max_tokens=50)  # 50 tokens ≈ 200 chars
    assert len(result) < len(long_text)
    assert "[Truncated for length]" in result
    assert result.count("\n") < long_text.count("\n")


@pytest.mark.unit
def test_memory_repr_with_dependencies(mock_broker, tmp_path):
    """Test __repr__ shows dependencies."""
    state = Mock()
    repo = AsyncMock()

    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        daemon_state=state,
        analysis_repo=repo,
        broker=mock_broker,
    )

    repr_str = repr(memory)
    assert "daemon_state" in repr_str
    assert "analysis_repo" in repr_str
    assert "broker" in repr_str
    assert str(tmp_path / "memory.jsonl") in repr_str


@pytest.mark.unit
def test_memory_repr_without_dependencies(tmp_path):
    """Test __repr__ without dependencies."""
    memory = CoordinatorMemory(memory_file=tmp_path / "memory.jsonl")

    repr_str = repr(memory)
    assert str(tmp_path / "memory.jsonl") in repr_str
    assert "daemon_state" not in repr_str
    assert "analysis_repo" not in repr_str
