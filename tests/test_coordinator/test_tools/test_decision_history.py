"""Tests for QueryPastDecisionsTool."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.v1.coordinator.decision_models import DecisionQueryResult
from src.v1.coordinator.tools.decision_history import QueryPastDecisionsTool


@pytest.fixture
def mock_coordinator_memory():
    """Create mock coordinator memory."""
    memory = MagicMock()
    memory.query_decisions = AsyncMock()
    memory.get_success_rate = AsyncMock()
    return memory


@pytest.fixture
def sample_decision_results():
    """Create sample decision query results."""
    base_time = datetime.now(UTC) - timedelta(days=30)

    return [
        DecisionQueryResult(
            symbol="AAPL",
            timestamp=base_time,
            signal="BUY",
            confidence=0.85,
            price_at_signal=150.0,
            price_at_outcome=155.0,
            return_pct=3.33,
            hit_miss="HIT",
            regime="trending_bullish",
            strategy_used="momentum",
            trading_session="REGULAR",
        ),
        DecisionQueryResult(
            symbol="MSFT",
            timestamp=base_time + timedelta(days=1),
            signal="BUY",
            confidence=0.80,
            price_at_signal=300.0,
            price_at_outcome=295.0,
            return_pct=-1.67,
            hit_miss="MISS",
            regime="trending_bullish",
            strategy_used="momentum",
            trading_session="REGULAR",
        ),
        DecisionQueryResult(
            symbol="GOOGL",
            timestamp=base_time + timedelta(days=2),
            signal="SELL",
            confidence=0.90,
            price_at_signal=140.0,
            price_at_outcome=135.0,
            return_pct=-3.57,
            hit_miss="HIT",
            regime="trending_bearish",
            strategy_used="momentum",
            trading_session="REGULAR",
        ),
    ]


@pytest.fixture
def sample_success_stats():
    """Create sample success rate statistics."""
    return {
        "total_decisions": 3,
        "hit_count": 2,
        "miss_count": 1,
        "pending_count": 0,
        "success_rate": 0.667,
        "avg_return": -0.30,
        "avg_confidence": 0.85,
    }


class TestQueryPastDecisionsTool:
    """Test QueryPastDecisionsTool."""

    def test_tool_name(self, mock_coordinator_memory):
        """Test tool name property."""
        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        assert tool.name == "query_past_decisions"

    def test_requires_confirmation(self, mock_coordinator_memory):
        """Test tool doesn't require confirmation (read-only)."""
        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        assert tool.requires_confirmation is False

    def test_get_tool_definition(self, mock_coordinator_memory):
        """Test tool definition structure."""
        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        definition = tool.get_tool_definition()

        assert definition.function.name == "query_past_decisions"
        assert "historical patterns" in definition.function.description.lower()
        assert "symbol" in definition.function.parameters.properties
        assert "signal" in definition.function.parameters.properties
        assert "lookback_days" in definition.function.parameters.properties
        assert definition.function.parameters.required == []

    def test_execute_basic(self, mock_coordinator_memory, sample_decision_results, sample_success_stats):
        """Test basic tool execution without filters."""
        mock_coordinator_memory.query_decisions.return_value = sample_decision_results
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        result = tool.execute()

        assert isinstance(result, str)
        assert "Past Trading Decisions" in result
        assert "Summary Statistics" in result
        assert "Success Rate:" in result
        assert "66.7%" in result  # 66.7% success rate

    def test_execute_with_symbol_filter(
        self, mock_coordinator_memory, sample_decision_results, sample_success_stats
    ):
        """Test execution with symbol filter."""
        aapl_decisions = [d for d in sample_decision_results if d.symbol == "AAPL"]
        mock_coordinator_memory.query_decisions.return_value = aapl_decisions
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        result = tool.execute(symbol="AAPL")

        assert "AAPL" in result
        mock_coordinator_memory.query_decisions.assert_called_once()
        call_params = mock_coordinator_memory.query_decisions.call_args[0][0]
        assert call_params.symbol == "AAPL"

    def test_execute_with_signal_filter(
        self, mock_coordinator_memory, sample_decision_results, sample_success_stats
    ):
        """Test execution with signal type filter."""
        buy_decisions = [d for d in sample_decision_results if d.signal == "BUY"]
        mock_coordinator_memory.query_decisions.return_value = buy_decisions
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        tool.execute(signal="BUY")

        mock_coordinator_memory.query_decisions.assert_called_once()
        call_params = mock_coordinator_memory.query_decisions.call_args[0][0]
        assert call_params.signal == "BUY"

    def test_execute_with_lookback_days(
        self, mock_coordinator_memory, sample_decision_results, sample_success_stats
    ):
        """Test execution with custom lookback period."""
        mock_coordinator_memory.query_decisions.return_value = sample_decision_results
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        result = tool.execute(lookback_days=30)

        assert "Last 30 Days" in result
        call_params = mock_coordinator_memory.query_decisions.call_args[0][0]
        assert call_params.lookback_days == 30

    def test_execute_with_lookback_days_capped(
        self, mock_coordinator_memory, sample_decision_results, sample_success_stats
    ):
        """Test lookback days is capped at maximum."""
        mock_coordinator_memory.query_decisions.return_value = sample_decision_results
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        tool.execute(lookback_days=500)  # Over max of 365

        call_params = mock_coordinator_memory.query_decisions.call_args[0][0]
        assert call_params.lookback_days == 365

    def test_execute_with_min_confidence(
        self, mock_coordinator_memory, sample_decision_results, sample_success_stats
    ):
        """Test execution with minimum confidence filter."""
        high_conf_decisions = [d for d in sample_decision_results if d.confidence >= 0.85]
        mock_coordinator_memory.query_decisions.return_value = high_conf_decisions
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        tool.execute(min_confidence=0.85)

        call_params = mock_coordinator_memory.query_decisions.call_args[0][0]
        assert call_params.min_confidence == 0.85

    def test_execute_with_horizon(
        self, mock_coordinator_memory, sample_decision_results, sample_success_stats
    ):
        """Test execution with different outcome horizon."""
        mock_coordinator_memory.query_decisions.return_value = sample_decision_results
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        result = tool.execute(horizon="20d")

        assert "20d Horizon" in result
        call_params = mock_coordinator_memory.query_decisions.call_args[0][0]
        assert call_params.horizon == "20d"

    def test_execute_with_invalid_horizon_defaults(
        self, mock_coordinator_memory, sample_decision_results, sample_success_stats
    ):
        """Test invalid horizon defaults to 5d."""
        mock_coordinator_memory.query_decisions.return_value = sample_decision_results
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        tool.execute(horizon="invalid")

        call_params = mock_coordinator_memory.query_decisions.call_args[0][0]
        assert call_params.horizon == "5d"

    def test_format_results_with_decisions(
        self, mock_coordinator_memory, sample_decision_results, sample_success_stats
    ):
        """Test result formatting includes table and statistics."""
        mock_coordinator_memory.query_decisions.return_value = sample_decision_results
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        result = tool.execute()

        # Check table headers
        assert "| Date | Symbol | Signal | Conf | Entry | Outcome | Return | Result |" in result

        # Check data rows
        assert "AAPL" in result
        assert "BUY" in result
        assert "✅ HIT" in result
        assert "❌ MISS" in result

        # Check summary stats
        assert "Success Rate:" in result
        assert "Total Decisions:" in result
        assert "Average Confidence:" in result

    def test_format_results_no_decisions(self, mock_coordinator_memory, sample_success_stats):
        """Test formatting when no decisions found."""
        mock_coordinator_memory.query_decisions.return_value = []
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        result = tool.execute(symbol="NONEXISTENT")

        assert "No decisions found matching filters" in result

    def test_format_results_with_pending_decisions(self, mock_coordinator_memory):
        """Test formatting includes pending decisions."""
        pending_decision = DecisionQueryResult(
            symbol="TSLA",
            timestamp=datetime.now(UTC),
            signal="BUY",
            confidence=0.75,
            price_at_signal=200.0,
            price_at_outcome=None,
            return_pct=None,
            hit_miss="PENDING",
            regime="ranging",
            strategy_used="momentum",
            trading_session="REGULAR",
        )

        mock_coordinator_memory.query_decisions.return_value = [pending_decision]
        mock_coordinator_memory.get_success_rate.return_value = {
            "total_decisions": 1,
            "hit_count": 0,
            "miss_count": 0,
            "pending_count": 1,
            "success_rate": 0.0,
            "avg_return": None,
            "avg_confidence": 0.75,
        }

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        result = tool.execute()

        assert "⏳ PENDING" in result
        assert "—" in result  # Placeholder for missing outcome

    def test_format_results_limits_rows(self, mock_coordinator_memory, sample_success_stats):
        """Test table limits to 30 rows for readability."""
        # Create 50 decisions
        many_decisions = [
            DecisionQueryResult(
                symbol=f"SYM{i}",
                timestamp=datetime.now(UTC) - timedelta(days=i),
                signal="BUY",
                confidence=0.8,
                price_at_signal=100.0,
                price_at_outcome=105.0,
                return_pct=5.0,
                hit_miss="HIT",
                regime="trending",
                strategy_used="momentum",
                trading_session="REGULAR",
            )
            for i in range(50)
        ]

        mock_coordinator_memory.query_decisions.return_value = many_decisions
        mock_coordinator_memory.get_success_rate.return_value = sample_success_stats

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        result = tool.execute()

        assert "Showing first 30 of 50 decisions" in result

    def test_execute_error_handling(self, mock_coordinator_memory):
        """Test error handling when query fails."""
        mock_coordinator_memory.query_decisions.side_effect = Exception("Database error")

        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        result = tool.execute()

        assert "Failed to query past decisions" in result
        assert "Database error" in result

    def test_repr(self, mock_coordinator_memory):
        """Test string representation."""
        tool = QueryPastDecisionsTool(mock_coordinator_memory)

        assert repr(tool) == "QueryPastDecisionsTool()"
