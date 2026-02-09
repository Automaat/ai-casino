"""Tests for portfolio optimization."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.broker import AlpacaBroker, BrokerAccountInfo, BrokerPosition
from src.data.market import MarketData, MarketDataFetcher
from src.optimization.portfolio import OptimizedPortfolio, PortfolioAllocation, PortfolioOptimizer


@pytest.fixture
def sample_returns_data() -> pd.DataFrame:
    """Generate sample returns data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="D")
    data = {
        "AAPL": [0.01, -0.005, 0.008, 0.003, -0.002] * 50 + [0.001, 0.002],
        "MSFT": [0.008, -0.003, 0.012, 0.001, -0.004] * 50 + [0.003, -0.001],
        "GOOGL": [-0.002, 0.010, 0.005, -0.001, 0.007] * 50 + [0.002, 0.005],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_price_data() -> dict[str, pd.DataFrame]:
    """Generate sample price data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="D")
    price_base = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 140.0}

    result = {}
    for symbol, base_price in price_base.items():
        prices = [base_price * (1 + i * 0.001) for i in range(252)]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.02 for p in prices],
                "low": [p * 0.98 for p in prices],
                "close": prices,
                "volume": [1000000] * 252,
            },
            index=dates,
        )
        result[symbol] = df

    return result


@pytest.fixture
def mock_market_fetcher(sample_price_data: dict[str, pd.DataFrame]) -> MarketDataFetcher:
    """Mock market data fetcher."""
    fetcher = MagicMock(spec=MarketDataFetcher)

    def fetch_daily(symbol: str, period_days: int = 90) -> MarketData:
        if symbol in sample_price_data:
            return MarketData(symbol=symbol, data=sample_price_data[symbol], last_updated=datetime.now())
        msg = f"No data for {symbol}"
        raise ValueError(msg)

    fetcher.fetch_daily = fetch_daily
    return fetcher


@pytest.fixture
def mock_alpaca_broker() -> AlpacaBroker:
    """Mock Alpaca broker with sample positions."""
    broker = MagicMock(spec=AlpacaBroker)

    positions = {
        "AAPL": BrokerPosition(
            symbol="AAPL",
            qty=10,
            market_value=1500.0,
            avg_entry_price=145.0,
            unrealized_pnl=50.0,
            unrealized_pnl_percent=0.034,
        ),
        "MSFT": BrokerPosition(
            symbol="MSFT",
            qty=5,
            market_value=1500.0,
            avg_entry_price=290.0,
            unrealized_pnl=50.0,
            unrealized_pnl_percent=0.034,
        ),
    }

    account_info = BrokerAccountInfo(
        balance=3000.0,
        available_cash=0.0,
        positions=positions,
        total_exposure=3000.0,
        portfolio_value=3000.0,
    )

    broker.get_account_info.return_value = account_info
    return broker


@pytest.fixture
def portfolio_optimizer(mock_market_fetcher: MarketDataFetcher) -> PortfolioOptimizer:
    """Portfolio optimizer without broker."""
    return PortfolioOptimizer(mock_market_fetcher, broker=None, period_days=252)


@pytest.fixture
def portfolio_optimizer_with_broker(
    mock_market_fetcher: MarketDataFetcher, mock_alpaca_broker: AlpacaBroker
) -> PortfolioOptimizer:
    """Portfolio optimizer with mocked broker."""
    return PortfolioOptimizer(mock_market_fetcher, broker=mock_alpaca_broker, period_days=252)


@pytest.mark.unit
def test_optimize_max_sharpe_basic(portfolio_optimizer: PortfolioOptimizer) -> None:
    """Test max Sharpe optimization returns valid portfolio."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    result = portfolio_optimizer.optimize_max_sharpe(symbols)

    assert isinstance(result, OptimizedPortfolio)
    assert result.method == "max_sharpe"
    assert 0.99 <= result.total_weight <= 1.01
    assert len(result.allocations) > 0
    assert all(0 <= a.weight <= 1 for a in result.allocations)


@pytest.mark.unit
def test_optimize_min_volatility_basic(portfolio_optimizer: PortfolioOptimizer) -> None:
    """Test min volatility optimization returns valid portfolio."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    result = portfolio_optimizer.optimize_min_volatility(symbols)

    assert isinstance(result, OptimizedPortfolio)
    assert result.method == "min_volatility"
    assert 0.99 <= result.total_weight <= 1.01
    assert len(result.allocations) > 0
    assert all(0 <= a.weight <= 1 for a in result.allocations)


@pytest.mark.unit
def test_optimize_hrp_basic(portfolio_optimizer: PortfolioOptimizer) -> None:
    """Test HRP optimization returns valid portfolio."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    result = portfolio_optimizer.optimize_hrp(symbols)

    assert isinstance(result, OptimizedPortfolio)
    assert result.method == "hrp"
    assert 0.99 <= result.total_weight <= 1.01
    assert len(result.allocations) > 0
    # HRP typically keeps all symbols
    assert len(result.allocations) >= 2


@pytest.mark.unit
def test_get_current_portfolio_from_alpaca(portfolio_optimizer_with_broker: PortfolioOptimizer) -> None:
    """Test fetching current portfolio from Alpaca."""
    weights = portfolio_optimizer_with_broker.get_current_portfolio()

    assert isinstance(weights, dict)
    assert "AAPL" in weights
    assert "MSFT" in weights
    assert weights["AAPL"] == 0.5  # 1500 / 3000
    assert weights["MSFT"] == 0.5  # 1500 / 3000
    assert sum(weights.values()) == 1.0


@pytest.mark.unit
def test_get_current_portfolio_no_broker_error(portfolio_optimizer: PortfolioOptimizer) -> None:
    """Test get_current_portfolio raises error when broker not configured."""
    with pytest.raises(ValueError, match="Broker not configured"):
        portfolio_optimizer.get_current_portfolio()


@pytest.mark.unit
def test_calculate_rebalance_with_manual_weights(portfolio_optimizer: PortfolioOptimizer) -> None:
    """Test rebalancing with manually provided current weights."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    target = portfolio_optimizer.optimize_max_sharpe(symbols)

    current = {"AAPL": 0.6, "MSFT": 0.4}
    rebalances = portfolio_optimizer.calculate_rebalance(target, current=current)

    assert isinstance(rebalances, list)
    assert len(rebalances) > 0
    assert all(hasattr(r, "symbol") for r in rebalances)
    assert all(hasattr(r, "action") for r in rebalances)
    assert all(r.action in ["BUY", "SELL", "HOLD"] for r in rebalances)

    # Check deltas sum close to 0 (conservation)
    total_delta = sum(r.delta for r in rebalances)
    assert float(abs(total_delta)) < 0.01


@pytest.mark.unit
def test_calculate_rebalance_auto_fetch_from_alpaca(
    portfolio_optimizer_with_broker: PortfolioOptimizer,
) -> None:
    """Test rebalancing auto-fetches from Alpaca when current=None."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    target = portfolio_optimizer_with_broker.optimize_max_sharpe(symbols)

    rebalances = portfolio_optimizer_with_broker.calculate_rebalance(target, current=None)

    assert isinstance(rebalances, list)
    assert len(rebalances) > 0
    # Should have fetched AAPL and MSFT from broker
    symbols_in_rebalance = {r.symbol for r in rebalances}
    assert "AAPL" in symbols_in_rebalance
    assert "MSFT" in symbols_in_rebalance


@pytest.mark.unit
def test_calculate_rebalance_includes_shares(
    portfolio_optimizer_with_broker: PortfolioOptimizer,
) -> None:
    """Test rebalancing calculates shares_to_trade when broker available."""
    symbols = ["AAPL", "MSFT"]
    target = portfolio_optimizer_with_broker.optimize_max_sharpe(symbols)

    rebalances = portfolio_optimizer_with_broker.calculate_rebalance(target, current=None)

    # Should have shares calculated for existing positions
    aapl_rebalance = next((r for r in rebalances if r.symbol == "AAPL"), None)
    assert aapl_rebalance is not None
    # shares_to_trade may be None if action is HOLD or 0
    if aapl_rebalance.action != "HOLD":
        assert isinstance(aapl_rebalance.shares_to_trade, int)


@pytest.mark.unit
def test_optimize_single_symbol_error(portfolio_optimizer: PortfolioOptimizer) -> None:
    """Test optimization fails with single symbol."""
    with pytest.raises(ValueError, match="at least 2 symbols"):
        portfolio_optimizer.optimize_max_sharpe(["AAPL"])


@pytest.mark.unit
def test_optimize_empty_data_error(mock_market_fetcher: MarketDataFetcher) -> None:
    """Test optimization fails with empty data."""

    # Mock fetcher to return empty data
    def fetch_empty(symbol: str, period_days: int = 90) -> MarketData:
        return MarketData(symbol=symbol, data=pd.DataFrame(), last_updated=datetime.now())

    mock_market_fetcher.fetch_daily = fetch_empty
    optimizer = PortfolioOptimizer(mock_market_fetcher)

    with pytest.raises(ValueError, match="Insufficient data"):
        optimizer.optimize_max_sharpe(["AAPL", "MSFT"])


@pytest.mark.unit
@pytest.mark.parametrize(
    "method_name",
    [
        "optimize_max_sharpe",
        "optimize_min_volatility",
        "optimize_hrp",
    ],
)
def test_all_methods_return_valid_portfolios(
    portfolio_optimizer: PortfolioOptimizer, method_name: str
) -> None:
    """Test all optimization methods return valid portfolios."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    method = getattr(portfolio_optimizer, method_name)
    result = method(symbols)

    assert isinstance(result, OptimizedPortfolio)
    assert 0.99 <= result.total_weight <= 1.01
    assert len(result.allocations) > 0
    assert all(isinstance(a, PortfolioAllocation) for a in result.allocations)
    assert isinstance(result.optimization_date, datetime)


@pytest.mark.unit
def test_portfolio_allocation_validation() -> None:
    """Test PortfolioAllocation Pydantic validation."""
    # Valid allocation
    alloc = PortfolioAllocation(symbol="AAPL", weight=0.5, expected_return=0.1, contribution_to_risk=0.2)
    assert alloc.symbol == "AAPL"
    assert alloc.weight == 0.5

    # Invalid weight (>1)
    with pytest.raises(ValueError, match="less than or equal to 1"):
        PortfolioAllocation(symbol="AAPL", weight=1.5)

    # Invalid weight (<0)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        PortfolioAllocation(symbol="AAPL", weight=-0.1)


@pytest.mark.unit
def test_rebalance_no_broker_no_current_error(portfolio_optimizer: PortfolioOptimizer) -> None:
    """Test rebalancing fails when no broker and no current weights."""
    symbols = ["AAPL", "MSFT"]
    target = portfolio_optimizer.optimize_max_sharpe(symbols)

    with pytest.raises(ValueError, match="No current portfolio provided"):
        portfolio_optimizer.calculate_rebalance(target, current=None)


@pytest.mark.unit
def test_optimize_with_insufficient_data_points(mock_market_fetcher: MarketDataFetcher) -> None:
    """Test optimization with <30 data points raises error."""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
    small_df = pd.DataFrame(
        {
            "open": [150.0] * 20,
            "high": [152.0] * 20,
            "low": [148.0] * 20,
            "close": [150.0] * 20,
            "volume": [1000000] * 20,
        },
        index=dates,
    )

    def fetch_small(symbol: str, period_days: int = 90) -> MarketData:
        return MarketData(symbol=symbol, data=small_df, last_updated=datetime.now())

    mock_market_fetcher.fetch_daily = fetch_small
    optimizer = PortfolioOptimizer(mock_market_fetcher, period_days=20)

    with pytest.raises(ValueError, match="Insufficient data points"):
        optimizer.optimize_max_sharpe(["AAPL", "MSFT"])


@pytest.mark.unit
def test_repr(portfolio_optimizer: PortfolioOptimizer) -> None:
    """Test __repr__ method."""
    repr_str = repr(portfolio_optimizer)
    assert "PortfolioOptimizer" in repr_str
    assert "period_days=252" in repr_str
    assert "broker=no" in repr_str
