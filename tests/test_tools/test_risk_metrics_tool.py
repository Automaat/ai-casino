"""Tests for GetRiskMetricsTool."""

from unittest.mock import MagicMock, patch

import pytest

from src.tools.risk_metrics import GetRiskMetricsTool


@pytest.fixture
def tool():
    """Create GetRiskMetricsTool."""
    return GetRiskMetricsTool()


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

    def test_name(self, tool):
        """Test tool name."""
        assert tool.name == "get_risk_metrics"

    def test_requires_confirmation(self, tool):
        """Test that tool doesn't require confirmation."""
        assert tool.requires_confirmation is False

    def test_get_tool_definition(self, tool):
        """Test tool definition format."""
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_risk_metrics"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "days" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, tool, mock_risk_metrics, mock_market_data):
        """Test successful execution."""
        with (
            patch("src.data.market.MarketDataFetcher") as mock_fetcher_cls,
            patch("src.metrics.risk.RiskMetricsCalculator") as mock_calc_cls,
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_daily.return_value = mock_market_data
            mock_fetcher_cls.return_value = mock_fetcher

            mock_calc = MagicMock()
            mock_calc.calculate_all.return_value = mock_risk_metrics
            mock_calc_cls.return_value = mock_calc

            result = tool.execute(symbol="AAPL", days=90)

            assert "AAPL" in result
            assert "0.0234" in result  # VaR 95
            assert "0.1523" in result  # Max drawdown
            assert "0.2345" in result  # Volatility
            mock_fetcher.fetch_daily.assert_called_once_with("AAPL", period_days=90)

    def test_execute_uppercase_symbol(self, tool, mock_risk_metrics, mock_market_data):
        """Test that symbol is uppercased."""
        with (
            patch("src.data.market.MarketDataFetcher") as mock_fetcher_cls,
            patch("src.metrics.risk.RiskMetricsCalculator") as mock_calc_cls,
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_daily.return_value = mock_market_data
            mock_fetcher_cls.return_value = mock_fetcher

            mock_calc = MagicMock()
            mock_calc.calculate_all.return_value = mock_risk_metrics
            mock_calc_cls.return_value = mock_calc

            tool.execute(symbol="aapl")

            mock_fetcher.fetch_daily.assert_called_once_with("AAPL", period_days=90)

    def test_execute_error_handling(self, tool):
        """Test error handling on failure."""
        with patch("src.data.market.MarketDataFetcher") as mock_fetcher_cls:
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_daily.side_effect = Exception("No data")
            mock_fetcher_cls.return_value = mock_fetcher

            result = tool.execute(symbol="INVALID")

            assert "Risk metrics calculation failed" in result
            assert "No data" in result

    def test_repr(self, tool):
        """Test string representation."""
        repr_str = repr(tool)
        assert "GetRiskMetricsTool" in repr_str
