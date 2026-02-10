"""Tests for GetRiskMetricsTool."""

from unittest.mock import MagicMock

import pytest

from src.tools.risk_metrics import GetRiskMetricsTool


@pytest.fixture
def mock_risk_metrics():
    """Create mock RiskMetrics."""
    var_metrics = MagicMock()
    var_metrics.var_95 = 0.0234
    var_metrics.var_99 = 0.0412
    var_metrics.cvar_95 = 0.0356
    var_metrics.cvar_99 = 0.0523

    dd_metrics = MagicMock()
    dd_metrics.max_drawdown = 0.1523
    dd_metrics.cdar_95 = 0.1234
    dd_metrics.avg_drawdown = 0.0456
    dd_metrics.max_drawdown_duration_days = 15

    metrics = MagicMock()
    metrics.var_metrics = var_metrics
    metrics.drawdown_metrics = dd_metrics
    metrics.volatility_annual = 0.2345
    metrics.downside_deviation = 0.1678
    return metrics


@pytest.fixture
def mock_market_data():
    """Create mock market data with close prices."""
    import pandas as pd

    data = MagicMock()
    close_series = pd.Series([100.0, 101.0, 99.5, 102.0, 101.5], name="Close")
    data.data = pd.DataFrame({"Close": close_series})
    return data


class TestGetRiskMetricsTool:
    """Tests for GetRiskMetricsTool."""

    def test_name(self, test_container_full):
        """Test tool name."""
        tool = GetRiskMetricsTool(container=test_container_full)
        assert tool.name == "get_risk_metrics"

    def test_requires_confirmation(self, test_container_full):
        """Test that tool doesn't require confirmation."""
        tool = GetRiskMetricsTool(container=test_container_full)
        assert tool.requires_confirmation is False

    def test_get_tool_definition(self, test_container_full):
        """Test tool definition format."""
        tool = GetRiskMetricsTool(container=test_container_full)
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_risk_metrics"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "days" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, test_container_full, mock_risk_metrics, mock_market_data):
        """Test successful execution."""
        from src.metrics.risk import RiskMetricsCalculator

        tool = GetRiskMetricsTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_daily.return_value = mock_market_data
        test_container_full.market_fetcher.override(mock_fetcher)

        mock_calc = MagicMock(spec=RiskMetricsCalculator)
        mock_calc.calculate_all.return_value = mock_risk_metrics
        test_container_full.risk_metrics_calculator.override(mock_calc)

        result = tool.execute(symbol="AAPL", days=90)

        assert "AAPL" in result
        assert "0.0234" in result  # VaR 95
        assert "0.1523" in result  # Max drawdown
        assert "0.2345" in result  # Volatility
        mock_fetcher.fetch_daily.assert_called_once_with("AAPL", period_days=90)

    def test_execute_uppercase_symbol(self, test_container_full, mock_risk_metrics, mock_market_data):
        """Test that symbol is uppercased."""
        from src.metrics.risk import RiskMetricsCalculator

        tool = GetRiskMetricsTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_daily.return_value = mock_market_data
        test_container_full.market_fetcher.override(mock_fetcher)

        mock_calc = MagicMock(spec=RiskMetricsCalculator)
        mock_calc.calculate_all.return_value = mock_risk_metrics
        test_container_full.risk_metrics_calculator.override(mock_calc)

        tool.execute(symbol="aapl")

        mock_fetcher.fetch_daily.assert_called_once_with("AAPL", period_days=90)

    def test_execute_error_handling(self, test_container_full):
        """Test error handling on failure."""
        tool = GetRiskMetricsTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_daily.side_effect = Exception("No data")
        test_container_full.market_fetcher.override(mock_fetcher)

        result = tool.execute(symbol="INVALID")

        assert "Risk metrics calculation failed" in result
        assert "No data" in result

    def test_repr(self, test_container_full):
        """Test string representation."""
        tool = GetRiskMetricsTool(container=test_container_full)
        repr_str = repr(tool)
        assert "GetRiskMetricsTool" in repr_str
