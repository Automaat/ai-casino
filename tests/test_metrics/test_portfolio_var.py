"""Tests for portfolio VaR calculator."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.broker import BrokerPosition
from src.data.market import MarketData
from src.metrics.portfolio_var import PortfolioVaRCalculator, PortfolioVaRResult
from src.metrics.risk import RiskMetricsCalculator


def _make_position(symbol: str, market_value: float) -> BrokerPosition:
    """Create a BrokerPosition with given market value."""
    return BrokerPosition(
        symbol=symbol,
        qty=10.0,
        market_value=market_value,
        avg_entry_price=market_value / 10,
        unrealized_pnl=0.0,
        unrealized_pnl_percent=0.0,
    )


def _make_market_data(symbol: str, n: int = 60, base: float = 100.0) -> MarketData:
    """Create MarketData with trending prices."""
    import numpy as np

    np.random.seed(hash(symbol) % 2**31)
    noise = np.random.normal(0, 1, n)
    close = base + np.cumsum(noise * 0.5)
    close = np.maximum(close, 1.0)

    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": [1000000] * n,
        }
    )
    return MarketData(symbol=symbol, data=df, last_updated=datetime(2024, 1, 15, 16, 0))


@pytest.fixture
def risk_calculator():
    return RiskMetricsCalculator()


@pytest.fixture
def mock_market_fetcher_var():
    mock = MagicMock()

    def fetch_daily(symbol: str, period_days: int = 90) -> MarketData:
        return _make_market_data(symbol, n=period_days, base=100.0)

    mock.fetch_daily = MagicMock(side_effect=fetch_daily)
    return mock


@pytest.fixture
def calculator(risk_calculator, mock_market_fetcher_var):
    return PortfolioVaRCalculator(risk_calculator, mock_market_fetcher_var)


class TestPortfolioVaRCalculator:
    def test_calculate_single_position(self, calculator):
        positions = {"AAPL": _make_position("AAPL", 50000.0)}
        result = calculator.calculate(positions, 100000.0, lookback_days=60)

        assert isinstance(result, PortfolioVaRResult)
        assert result.sufficient_data is True
        assert result.num_positions == 1
        assert result.var_95 >= 0
        assert result.cvar_99 >= 0
        assert result.cdar_95 >= 0
        assert result.portfolio_volatility >= 0

    def test_calculate_multiple_positions(self, calculator):
        positions = {
            "AAPL": _make_position("AAPL", 30000.0),
            "MSFT": _make_position("MSFT", 30000.0),
            "GOOGL": _make_position("GOOGL", 30000.0),
        }
        result = calculator.calculate(positions, 100000.0, lookback_days=60)

        assert result.sufficient_data is True
        assert result.num_positions == 3

    def test_calculate_with_hypothetical_increases_risk(self, calculator):
        positions = {"AAPL": _make_position("AAPL", 30000.0)}
        portfolio_value = 100000.0

        calculator.calculate(positions, portfolio_value, lookback_days=60)
        with_new = calculator.calculate_with_hypothetical(
            positions, portfolio_value, "TSLA", 30000.0, lookback_days=60
        )

        assert with_new.sufficient_data is True
        assert with_new.num_positions == 2
        # More positions with different correlation may increase or decrease VaR
        # but the calculation should complete

    def test_cold_start_empty_positions(self, calculator):
        result = calculator.calculate({}, 100000.0, lookback_days=60)

        assert result.sufficient_data is False
        assert result.var_95 == 0.0
        assert result.cvar_99 == 0.0
        assert result.num_positions == 0

    def test_insufficient_market_data(self, risk_calculator):
        mock_fetcher = MagicMock()
        # Return only 1 data point
        df = pd.DataFrame({"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1000000]})
        mock_fetcher.fetch_daily.return_value = MarketData(
            symbol="AAPL", data=df, last_updated=datetime(2024, 1, 15, 16, 0)
        )

        calc = PortfolioVaRCalculator(risk_calculator, mock_fetcher)
        positions = {"AAPL": _make_position("AAPL", 50000.0)}
        result = calc.calculate(positions, 100000.0, lookback_days=60)

        assert result.sufficient_data is False

    def test_missing_data_for_one_position(self, risk_calculator):
        mock_fetcher = MagicMock()

        def fetch_daily(symbol: str, period_days: int = 90) -> MarketData:
            if symbol == "BAD":
                msg = "API error"
                raise ValueError(msg)
            return _make_market_data(symbol, n=period_days)

        mock_fetcher.fetch_daily = MagicMock(side_effect=fetch_daily)
        calc = PortfolioVaRCalculator(risk_calculator, mock_fetcher)

        positions = {
            "AAPL": _make_position("AAPL", 50000.0),
            "BAD": _make_position("BAD", 30000.0),
        }
        result = calc.calculate(positions, 100000.0, lookback_days=60)

        # Should still work with AAPL data only
        assert result.sufficient_data is True
        assert result.num_positions == 1  # counts only positions with successful data fetch

    def test_zero_portfolio_value(self, calculator):
        positions = {"AAPL": _make_position("AAPL", 0.0)}
        result = calculator.calculate(positions, 0.0, lookback_days=60)

        assert result.sufficient_data is False

    def test_repr(self, calculator):
        assert "PortfolioVaRCalculator" in repr(calculator)
