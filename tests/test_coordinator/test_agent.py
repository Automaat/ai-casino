"""Tests for TradingCoordinator agent."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from src.v1.coordinator.agent import TradingCoordinator
from src.v1.coordinator.memory import ObservationRecord
from src.v1.coordinator.models import CoordinatorConfig, CoordinatorCycleResult


@pytest.fixture
def coordinator_config():
    """Create test coordinator config."""
    return CoordinatorConfig(
        enabled=True,
        max_tool_calls=10,
        temperature=0.5,
        confirmation_mode="auto",
        cycle_timeout_seconds=60,
        max_daily_trades=5,
        max_position_pct=10.0,
        min_confidence_to_trade=0.6,
    )


@pytest.fixture
def mock_llm():
    """Create mock LLM client."""
    mock = AsyncMock()
    mock.acomplete_with_tools = AsyncMock(
        return_value="Cycle complete. Analyzed 2 symbols, executed 1 trade."
    )
    return mock


@pytest.fixture
def mock_tool_registry():
    """Create mock tool registry."""
    mock = Mock()
    mock.get_definitions = Mock(return_value=[])
    mock.aexecute = AsyncMock(return_value="Tool executed")
    mock.requires_confirmation = Mock(return_value=False)
    mock.__len__ = Mock(return_value=9)
    return mock


@pytest.fixture
def mock_memory():
    """Create mock coordinator memory."""
    mock = AsyncMock()
    mock.retrieve_recent = AsyncMock(return_value=[])
    mock.get_today_summary = AsyncMock(return_value="No analyses today")
    mock.get_today_game_plan = AsyncMock(return_value="Game plan unavailable")
    mock.get_portfolio_summary = AsyncMock(
        return_value=(
            "## Current Portfolio\n"
            "- **Balance**: $10,000.00\n"
            "- **Portfolio Value**: $10,000.00\n"
            "- **Available Cash**: $10,000.00\n"
            "- **Total Exposure**: $0.00 (0.0%)"
        )
    )
    mock.query_decisions = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_broker():
    """Create mock broker."""
    from result import Ok

    from src.data.broker import BrokerAccountInfo

    mock = Mock()
    mock.get_account_info = Mock(
        return_value=Ok(
            BrokerAccountInfo(
                balance=10000.0,
                portfolio_value=12000.0,
                available_cash=8000.0,
                total_exposure=2000.0,
                positions={},
            )
        )
    )
    return mock


@pytest.fixture
def mock_critic_agent():
    """Create mock critic agent."""
    return AsyncMock()


@pytest.fixture
def coordinator(
    mock_llm, mock_tool_registry, mock_memory, coordinator_config, mock_broker, mock_critic_agent
):
    """Create TradingCoordinator instance."""
    return TradingCoordinator(
        llm_client=mock_llm,
        tool_registry=mock_tool_registry,
        memory=mock_memory,
        config=coordinator_config,
        broker=mock_broker,
        critic_agent=mock_critic_agent,
    )


@pytest.mark.asyncio
async def test_run_cycle_success(coordinator, mock_llm):
    """Test successful cycle execution."""
    result = await coordinator.run_cycle(watchlist=["AAPL", "TSLA"])

    assert isinstance(result, CoordinatorCycleResult)
    assert result.summary != ""
    assert result.tool_calls_made >= 0
    mock_llm.acomplete_with_tools.assert_called_once()


@pytest.mark.asyncio
async def test_run_cycle_tracks_tool_calls(coordinator, mock_llm):
    """Test tool call tracking."""

    # Simulate tool calls via callback
    def tool_call_side_effect(params):
        callback = params.on_tool_call
        if callback:
            # Simulate analyze_symbol call
            callback("analyze_symbol", {"symbol": "AAPL"}, "Analysis result")
            # Simulate execute_trade call
            callback("execute_trade", {"symbol": "AAPL", "action": "BUY"}, "Trade executed successfully")
        return "Cycle complete"

    mock_llm.acomplete_with_tools.side_effect = tool_call_side_effect

    result = await coordinator.run_cycle(watchlist=["AAPL"])

    assert result.tool_calls_made == 2
    assert "AAPL" in result.symbols_analyzed
    assert result.trades_proposed == 1
    assert result.trades_executed == 1


@pytest.mark.asyncio
async def test_run_cycle_handles_degradation(coordinator, mock_memory):
    """Test degradation context handling."""
    degradation_context = {
        "disabled_agents": ["news"],
        "message": "News API degraded",
    }

    result = await coordinator.run_cycle(
        watchlist=["AAPL"],
        degradation_context=degradation_context,
    )

    assert isinstance(result, CoordinatorCycleResult)
    # System prompt should have been built with degradation context
    assert mock_memory.retrieve_recent.called


@pytest.mark.asyncio
async def test_run_cycle_error_handling(coordinator, mock_llm):
    """Test error handling during cycle."""
    mock_llm.acomplete_with_tools.side_effect = ValueError("LLM error")

    result = await coordinator.run_cycle(watchlist=["AAPL"])

    assert isinstance(result, CoordinatorCycleResult)
    assert "Error" in result.summary


@pytest.mark.asyncio
async def test_run_cycle_timeout(coordinator, mock_llm):
    """Test cycle timeout handling."""
    import asyncio

    async def slow_llm(*args, **kwargs):
        await asyncio.sleep(100)
        return "Never completes"

    mock_llm.acomplete_with_tools.side_effect = slow_llm

    result = await coordinator.run_cycle(watchlist=["AAPL"])

    assert isinstance(result, CoordinatorCycleResult)
    assert "timeout" in result.summary.lower()


@pytest.mark.asyncio
async def test_tool_executor_confirmation(coordinator, mock_tool_registry, coordinator_config):
    """Test tool confirmation logic."""
    mock_tool_registry.requires_confirmation.return_value = True
    coordinator._config.confirmation_mode = "manual"

    result = await coordinator._tool_executor("execute_trade", {"symbol": "AAPL"})

    # In manual confirmation mode, execution should be deferred/awaiting confirmation
    assert "await" in str(result).lower()
    mock_tool_registry.aexecute.assert_not_called()


@pytest.mark.asyncio
async def test_tool_executor_error_handling(coordinator, mock_tool_registry):
    """Test tool executor error handling."""
    mock_tool_registry.aexecute.side_effect = ValueError("Tool error")

    result = await coordinator._tool_executor("broken_tool", {})

    assert "Error" in result


def test_on_tool_call_tracks_symbols(coordinator):
    """Test symbol tracking in callback."""
    coordinator._on_tool_call("analyze_symbol", {"symbol": "AAPL"}, "result")
    coordinator._on_tool_call("analyze_symbol", {"symbol": "TSLA"}, "result")

    assert coordinator._symbols_analyzed == {"AAPL", "TSLA"}
    assert coordinator._tool_calls_count == 2


def test_on_tool_call_tracks_trades(coordinator):
    """Test trade tracking in callback."""
    coordinator._on_tool_call("execute_trade", {"symbol": "AAPL"}, "Trade executed successfully")
    coordinator._on_tool_call("execute_trade", {"symbol": "TSLA"}, "Trade failed")

    assert coordinator._trades_proposed == 2
    assert coordinator._trades_executed == 1


@pytest.mark.asyncio
async def test_parse_cycle_result_success(coordinator):
    """Test cycle result parsing."""
    final_response = "Analyzed AAPL and TSLA. Executed 1 trade.\n\nNext cycle: monitor positions."

    result = await coordinator._parse_cycle_result(final_response)

    assert isinstance(result, CoordinatorCycleResult)
    assert result.summary == "Analyzed AAPL and TSLA. Executed 1 trade."


@pytest.mark.asyncio
async def test_parse_cycle_result_long_response(coordinator):
    """Test cycle result parsing with long response."""
    final_response = "A" * 300

    result = await coordinator._parse_cycle_result(final_response)

    assert len(result.summary) <= 203  # 200 chars + "..."


def test_format_memory_empty(coordinator):
    """Test memory formatting with no observations."""
    result = coordinator._format_memory([])

    assert result == ""


def test_format_memory_with_observations(coordinator):
    """Test memory formatting with observations."""
    observations = [
        ObservationRecord(
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            observation="AAPL broke resistance",
            category="pattern",
        ),
        ObservationRecord(
            timestamp=datetime(2024, 1, 2, 13, 0, 0, tzinfo=UTC),
            observation="Market trending up",
            category="market",
        ),
    ]

    result = coordinator._format_memory(observations)

    assert "Recent Observations" in result
    assert "pattern" in result
    assert "AAPL broke resistance" in result
    assert "market" in result


def test_format_risk_limits(coordinator):
    """Test risk limits formatting."""
    result = coordinator._format_risk_limits()

    assert "Risk Limits" in result
    assert "10.0%" in result
    assert "5" in result
    assert "60%" in result
    assert "auto" in result


def test_format_degradation_context_full(coordinator):
    """Test degradation context formatting."""
    context = {
        "disabled_agents": ["news", "social"],
        "degraded_tools": ["analyze_symbol"],
        "message": "High latency detected",
    }

    result = coordinator._format_degradation_context(context)

    assert "Degradation Warnings" in result
    assert "news, social" in result
    assert "analyze_symbol" in result
    assert "High latency detected" in result


def test_format_degradation_context_partial(coordinator):
    """Test degradation context with missing fields."""
    context = {"message": "Warning only"}

    result = coordinator._format_degradation_context(context)

    assert "Degradation Warnings" in result
    assert "Warning only" in result


@pytest.mark.asyncio
async def test_build_system_prompt_complete(coordinator, mock_memory, mock_broker):
    """Test system prompt building with all sections."""
    mock_memory.retrieve_recent.return_value = [
        ObservationRecord(
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            observation="Test observation",
            category="general",
        )
    ]

    degradation_context = {"message": "Test degradation"}

    result = await coordinator._build_system_prompt(["AAPL", "TSLA"], degradation_context)

    assert "trading coordinator" in result.lower()
    assert "Test observation" in result
    assert "Test degradation" in result
    assert "$10,000.00" in result
    assert "AAPL, TSLA" in result


@pytest.mark.asyncio
async def test_build_cycle_prompt(coordinator):
    """Test cycle prompt building."""
    result = await coordinator._build_cycle_prompt(["AAPL", "TSLA", "NVDA"])

    assert "AAPL, TSLA, NVDA" in result
    assert "No previous cycle" in result
    assert "No open positions" in result


@pytest.mark.asyncio
async def test_build_cycle_prompt_with_previous_summary(coordinator):
    """Test cycle prompt with previous summary."""
    coordinator._last_cycle_summary = "Previous cycle executed 2 trades"

    result = await coordinator._build_cycle_prompt(["AAPL"])

    assert "Previous cycle executed 2 trades" in result


def test_get_trading_mode_auto(coordinator):
    """Test trading mode for auto confirmation."""
    mode = coordinator._get_trading_mode()
    assert "AUTO" in mode
    assert "automatically" in mode


def test_get_trading_mode_manual(coordinator):
    """Test trading mode for manual confirmation."""
    coordinator._config.confirmation_mode = "manual"
    mode = coordinator._get_trading_mode()
    assert "MANUAL" in mode
    assert "confirmation" in mode


@pytest.mark.asyncio
async def test_get_positions_summary_no_positions(coordinator, mock_broker):
    """Test positions summary with no open positions."""
    from result import Ok

    from src.data.broker import BrokerAccountInfo

    mock_broker.get_account_info.return_value = Ok(
        BrokerAccountInfo(
            balance=10000.0,
            portfolio_value=12000.0,
            available_cash=8000.0,
            total_exposure=2000.0,
            positions={},
        )
    )
    summary = await coordinator._get_positions_summary()
    assert "No open positions" in summary


@pytest.mark.asyncio
async def test_get_positions_summary_with_positions(coordinator, mock_broker):
    """Test positions summary with open positions."""
    from result import Ok

    from src.data.broker import BrokerAccountInfo, BrokerPosition

    mock_broker.get_account_info.return_value = Ok(
        BrokerAccountInfo(
            balance=10000.0,
            portfolio_value=12000.0,
            available_cash=8000.0,
            total_exposure=2000.0,
            positions={
                "AAPL": BrokerPosition(
                    symbol="AAPL",
                    qty=10.0,
                    market_value=1500.0,
                    avg_entry_price=150.0,
                    unrealized_pnl=100.0,
                    unrealized_pnl_percent=6.67,
                )
            },
        )
    )

    summary = await coordinator._get_positions_summary()
    assert "1 open position:" in summary
    assert "AAPL" in summary
    assert "10" in summary
    assert "$150.00" in summary
    assert "profit" in summary


@pytest.mark.asyncio
async def test_get_positions_summary_error(coordinator, mock_broker):
    """Test positions summary error handling."""
    from result import Err

    mock_broker.get_account_info.return_value = Err(ValueError("Broker error"))
    summary = await coordinator._get_positions_summary()
    assert "unavailable" in summary


@pytest.mark.asyncio
async def test_build_cycle_prompt_includes_date_and_session(coordinator):
    """Test cycle prompt includes date and session context."""
    from src.strategies.session import TradingSession

    prompt = await coordinator._build_cycle_prompt(["AAPL", "TSLA"], TradingSession.PRE_MARKET)
    assert "PRE_MARKET" in prompt
    assert datetime.now(UTC).strftime("%Y-%m-%d") in prompt


@pytest.mark.asyncio
async def test_get_recent_outcomes_summary_empty(coordinator, mock_memory):
    """Test outcomes summary with no decisions."""
    mock_memory.query_decisions.return_value = []
    result = await coordinator._get_recent_outcomes_summary()
    assert result == ""


@pytest.mark.asyncio
async def test_get_recent_outcomes_summary_with_decisions(coordinator, mock_memory):
    """Test outcomes summary with decisions."""
    from src.v1.coordinator.decision_models import DecisionQueryResult

    mock_memory.query_decisions.return_value = [
        DecisionQueryResult(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            signal="BUY",
            confidence=0.8,
            price_at_signal=150.0,
            price_at_outcome=160.0,
            return_pct=6.7,
            hit_miss="HIT",
            regime="bullish",
            strategy_used="momentum",
            trading_session="REGULAR",
        ),
        DecisionQueryResult(
            symbol="TSLA",
            timestamp=datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
            signal="BUY",
            confidence=0.7,
            price_at_signal=200.0,
            price_at_outcome=190.0,
            return_pct=-5.0,
            hit_miss="MISS",
            regime="bullish",
            strategy_used="momentum",
            trading_session="REGULAR",
        ),
    ]

    result = await coordinator._get_recent_outcomes_summary()

    assert "Recent Trade Outcomes" in result
    assert "1/2 (50%)" in result
    assert "AAPL BUY 80% → HIT (+6.7%)" in result
    assert "TSLA BUY 70% → MISS (-5.0%)" in result


@pytest.mark.asyncio
async def test_get_recent_outcomes_summary_error(coordinator, mock_memory):
    """Test outcomes summary gracefully handles errors."""
    mock_memory.query_decisions.side_effect = RuntimeError("DB error")
    result = await coordinator._get_recent_outcomes_summary()
    assert result == ""


def test_repr(coordinator):
    """Test string representation."""
    result = repr(coordinator)

    assert "TradingCoordinator" in result
    assert "tools=9" in result
    assert "max_tool_calls=10" in result
    assert "confirmation=auto" in result
