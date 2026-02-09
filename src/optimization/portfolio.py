"""Portfolio optimization using scipy."""

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from scipy.optimize import minimize

from src.data.market import MarketDataFetcher

if TYPE_CHECKING:
    from src.data.broker import AlpacaBroker

# Constants
MIN_WEIGHT_THRESHOLD = 0.001  # Minimum weight to include in portfolio (0.1%)
REBALANCE_THRESHOLD = 0.01  # Minimum weight delta to trigger rebalance (1%)
MIN_SYMBOLS = 2  # Minimum symbols required for optimization
MIN_DATA_POINTS = 30  # Minimum historical data points required
WARN_DATA_POINTS = 50  # Data points threshold for warning


class PortfolioAllocation(BaseModel):
    """Individual asset allocation in portfolio."""

    symbol: str = Field(description="Stock ticker symbol")
    weight: float = Field(description="Portfolio weight (0-1)", ge=0.0, le=1.0)
    expected_return: float | None = Field(default=None, description="Expected annual return")
    contribution_to_risk: float | None = Field(default=None, description="Contribution to portfolio risk")


class OptimizedPortfolio(BaseModel):
    """Optimized portfolio result."""

    allocations: list[PortfolioAllocation] = Field(description="Asset allocations")
    expected_return: float = Field(description="Expected annual return")
    expected_volatility: float = Field(description="Expected annual volatility")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    method: str = Field(description="Optimization method used")
    optimization_date: datetime = Field(description="Optimization timestamp")

    @property
    def total_weight(self) -> float:
        """Total portfolio weight (should sum to 1.0)."""
        return sum(a.weight for a in self.allocations)


class PortfolioRebalance(BaseModel):
    """Rebalancing instruction for single asset."""

    symbol: str = Field(description="Stock ticker symbol")
    current_weight: float = Field(description="Current portfolio weight")
    target_weight: float = Field(description="Target portfolio weight")
    delta: float = Field(description="Weight delta (target - current)")
    action: str = Field(description="BUY, SELL, or HOLD")
    shares_to_trade: int | None = Field(default=None, description="Shares to buy/sell if broker available")


def _raise_optimization_error(message: str) -> None:
    """Raise optimization error with formatted message."""
    msg = f"Optimization failed: {message}"
    raise ValueError(msg)


class PortfolioOptimizer:
    """Portfolio optimizer using modern portfolio theory."""

    def __init__(
        self,
        market_fetcher: MarketDataFetcher,
        broker: "AlpacaBroker | None" = None,
        period_days: int = 365,
    ) -> None:
        """Initialize portfolio optimizer.

        Args:
            market_fetcher: Market data fetcher
            broker: Optional Alpaca broker for current portfolio
            period_days: Historical data period
        """
        self.market_fetcher = market_fetcher
        self.broker = broker
        self.period_days = period_days
        logger.info(
            f"Initialized PortfolioOptimizer (period={period_days} days, broker={'yes' if broker else 'no'})"
        )

    def optimize_max_sharpe(self, symbols: list[str]) -> OptimizedPortfolio:
        """Optimize portfolio for maximum Sharpe ratio.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Optimized portfolio
        """
        logger.info(f"Optimizing max Sharpe for {len(symbols)} symbols")
        returns_df = self._fetch_returns_data(symbols)

        # Initialize variables for type checker
        cleaned_weights: dict[str, float]
        expected_return: float
        volatility: float
        sharpe: float

        try:
            # Calculate mean returns and covariance
            mu = returns_df.mean() * 252  # Annualize
            cov = returns_df.cov() * 252

            n = len(symbols)
            rf = 0.02

            # Objective: minimize negative Sharpe ratio
            def neg_sharpe(weights: np.ndarray) -> float:
                ret = np.dot(weights, mu)
                vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
                return -(ret - rf) / vol if vol > 0 else 0.0

            # Constraints: weights sum to 1
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            bounds = tuple((0, 1) for _ in range(n))
            initial = np.array([1.0 / n] * n)

            result = minimize(neg_sharpe, initial, method="SLSQP", bounds=bounds, constraints=constraints)

            if result is not None and result.success:
                weights = result.x
                expected_return = np.dot(weights, mu)
                volatility = np.sqrt(np.dot(weights, np.dot(cov, weights)))
                sharpe = (expected_return - rf) / volatility if volatility > 0 else 0.0

                cleaned_weights = dict(zip(symbols, weights, strict=True))
            else:
                message = result.message if result is not None else "Optimization returned None"
                _raise_optimization_error(message)

        except Exception as e:
            logger.warning(f"Optimization failed, falling back to equal weights: {e}")
            cleaned_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
            expected_return, volatility, sharpe = 0.0, 0.0, 0.0

        allocations = [
            PortfolioAllocation(symbol=symbol, weight=weight, expected_return=None, contribution_to_risk=None)
            for symbol, weight in cleaned_weights.items()
            if weight > MIN_WEIGHT_THRESHOLD
        ]

        return OptimizedPortfolio(
            allocations=allocations,
            expected_return=expected_return,
            expected_volatility=volatility,
            sharpe_ratio=sharpe,
            method="max_sharpe",
            optimization_date=datetime.now(tz=UTC),
        )

    def optimize_min_volatility(self, symbols: list[str]) -> OptimizedPortfolio:
        """Optimize portfolio for minimum volatility.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Optimized portfolio
        """
        logger.info(f"Optimizing min volatility for {len(symbols)} symbols")
        returns_df = self._fetch_returns_data(symbols)

        # Initialize variables for type checker
        cleaned_weights: dict[str, float]
        expected_return: float
        volatility: float
        sharpe: float

        try:
            # Calculate mean returns and covariance
            mu = returns_df.mean() * 252
            cov = returns_df.cov() * 252

            n = len(symbols)
            rf = 0.02

            # Objective: minimize volatility
            def portfolio_volatility(weights: np.ndarray) -> float:
                return np.sqrt(np.dot(weights, np.dot(cov, weights)))

            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            bounds = tuple((0, 1) for _ in range(n))
            initial = np.array([1.0 / n] * n)

            result = minimize(
                portfolio_volatility, initial, method="SLSQP", bounds=bounds, constraints=constraints
            )

            if result is not None and result.success:
                weights = result.x
                expected_return = np.dot(weights, mu)
                volatility = np.sqrt(np.dot(weights, np.dot(cov, weights)))
                sharpe = (expected_return - rf) / volatility if volatility > 0 else 0.0

                cleaned_weights = dict(zip(symbols, weights, strict=True))
            else:
                message = result.message if result is not None else "Optimization returned None"
                _raise_optimization_error(message)

        except Exception as e:
            logger.warning(f"Optimization failed, falling back to equal weights: {e}")
            cleaned_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
            expected_return, volatility, sharpe = 0.0, 0.0, 0.0

        allocations = [
            PortfolioAllocation(symbol=symbol, weight=weight, expected_return=None, contribution_to_risk=None)
            for symbol, weight in cleaned_weights.items()
            if weight > MIN_WEIGHT_THRESHOLD
        ]

        return OptimizedPortfolio(
            allocations=allocations,
            expected_return=expected_return,
            expected_volatility=volatility,
            sharpe_ratio=sharpe,
            method="min_volatility",
            optimization_date=datetime.now(tz=UTC),
        )

    def optimize_hrp(self, symbols: list[str]) -> OptimizedPortfolio:
        """Optimize portfolio using Hierarchical Risk Parity.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Optimized portfolio
        """
        logger.info(f"Optimizing HRP for {len(symbols)} symbols")
        returns_df = self._fetch_returns_data(symbols)

        try:
            # Quasi-diagonalization (simplified HRP)
            n = len(symbols)

            # Simple inverse variance weighting as proxy for HRP
            cov = returns_df.cov() * 252
            var = np.diag(cov)

            # Guard against zero, negative, or non-finite variances
            valid = np.isfinite(var) & (var > 0)
            if not np.any(valid):
                logger.warning("HRP optimization: no valid variances found; using equal weights.")
                weights = np.ones(n) / n
            else:
                eps = 1e-8
                inv_var = np.zeros_like(var, dtype=float)
                inv_var[valid] = 1.0 / np.maximum(var[valid], eps)

                total_inv_var = inv_var.sum()
                if total_inv_var <= 0 or not np.isfinite(total_inv_var):
                    logger.warning("HRP optimization: invalid inverse-variance sum; using equal weights.")
                    weights = np.ones(n) / n
                else:
                    weights = inv_var / total_inv_var

            # Calculate metrics
            mu = returns_df.mean() * 252
            expected_return = np.dot(weights, mu)
            volatility = np.sqrt(np.dot(weights, np.dot(cov, weights)))
            sharpe = (expected_return - 0.02) / volatility if volatility > 0 else 0.0

            weights_dict = dict(zip(symbols, weights, strict=True))

        except Exception as e:
            logger.warning(f"HRP optimization failed, falling back to equal weights: {e}")
            weights_dict = {symbol: 1.0 / len(symbols) for symbol in symbols}
            expected_return, volatility, sharpe = 0.0, 0.0, 0.0

        allocations = [
            PortfolioAllocation(symbol=symbol, weight=weight, expected_return=None, contribution_to_risk=None)
            for symbol, weight in weights_dict.items()
            if weight > MIN_WEIGHT_THRESHOLD
        ]

        return OptimizedPortfolio(
            allocations=allocations,
            expected_return=expected_return,
            expected_volatility=volatility,
            sharpe_ratio=sharpe,
            method="hrp",
            optimization_date=datetime.now(tz=UTC),
        )

    def get_current_portfolio(self) -> dict[str, float]:
        """Fetch current portfolio weights from Alpaca broker.

        Returns:
            Dict of {symbol: weight} where weights sum to 1.0

        Raises:
            ValueError: If broker not configured
        """
        if not self.broker:
            err_msg = "Broker not configured - cannot fetch current portfolio"
            raise ValueError(err_msg)

        account_info = self.broker.get_account_info()
        portfolio_value = account_info.portfolio_value

        if portfolio_value == 0:
            logger.warning("Portfolio value is 0")
            return {}

        weights = {}
        for symbol, position in account_info.positions.items():
            weights[symbol] = position.market_value / portfolio_value

        logger.info(f"Fetched current portfolio: {len(weights)} positions")
        return weights

    def calculate_rebalance(
        self,
        target: OptimizedPortfolio,
        current: dict[str, float] | None = None,
        threshold: float = REBALANCE_THRESHOLD,
    ) -> list[PortfolioRebalance]:
        """Calculate rebalancing instructions.

        Args:
            target: Target optimized portfolio
            current: Current portfolio weights, auto-fetched if None
            threshold: Minimum weight delta to trigger rebalance (default 1%)

        Returns:
            List of rebalancing instructions sorted by abs(delta)
        """
        if current is None:
            if not self.broker:
                err_msg = "No current portfolio provided and broker not configured"
                raise ValueError(err_msg)
            current = self.get_current_portfolio()

        logger.info(
            f"Calculating rebalance: {len(current)} current → {len(target.allocations)} target positions"
        )

        # Build rebalance list
        rebalances = []
        all_symbols = set(current.keys()) | {a.symbol for a in target.allocations}

        for symbol in all_symbols:
            current_weight = current.get(symbol, 0.0)
            target_weight = next((a.weight for a in target.allocations if a.symbol == symbol), 0.0)
            delta = target_weight - current_weight

            action = ("BUY" if delta > 0 else "SELL") if abs(delta) > threshold else "HOLD"

            rebalances.append(
                PortfolioRebalance(
                    symbol=symbol,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    delta=delta,
                    action=action,
                    shares_to_trade=None,
                )
            )

        # Calculate shares from broker positions if available
        if self.broker:
            self._calculate_shares_to_trade(rebalances)

        # Sort by absolute delta descending
        rebalances.sort(key=lambda r: abs(r.delta), reverse=True)
        return rebalances

    def _calculate_shares_to_trade(self, rebalances: list[PortfolioRebalance]) -> None:
        """Calculate shares to trade for rebalancing (mutates rebalances list)."""
        if self.broker is None:
            logger.warning("Broker not configured, cannot calculate shares to trade")
            return

        account_info = self.broker.get_account_info()
        portfolio_value = account_info.portfolio_value

        for rebalance in rebalances:
            symbol = rebalance.symbol
            if symbol not in account_info.positions:
                # New position: calculate shares based on target weight
                if rebalance.action == "BUY":
                    try:
                        market_data = self.market_fetcher.fetch_daily(symbol, period_days=1)
                        latest_price = market_data.data["close"].iloc[-1]
                        if latest_price and latest_price > 0:
                            target_value = rebalance.target_weight * portfolio_value
                            rebalance.shares_to_trade = int(target_value / latest_price)
                            logger.debug(
                                f"New position {symbol}: {rebalance.shares_to_trade} shares "
                                f"@ ${latest_price:.2f} = ${target_value:.2f}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to fetch latest price for {symbol}: {e}")
                continue

            position = account_info.positions[symbol]
            current_price = position.market_value / position.qty if position.qty > 0 else 0
            if current_price <= 0:
                continue

            dollar_delta = rebalance.delta * portfolio_value
            shares = dollar_delta / current_price

            # Sign-preserving rounding: floor for sells, ceil for buys
            if shares < 0:
                rebalance.shares_to_trade = math.floor(shares)
            elif shares > 0:
                rebalance.shares_to_trade = math.ceil(shares)
            else:
                rebalance.shares_to_trade = 0

            # Validate action matches share direction
            self._validate_rebalance_action(rebalance, symbol)

    def _validate_rebalance_action(self, rebalance: PortfolioRebalance, symbol: str) -> None:
        """Validate rebalance action matches share direction (mutates rebalance)."""
        if rebalance.shares_to_trade is None or rebalance.shares_to_trade == 0:
            rebalance.action = "HOLD"
        elif rebalance.shares_to_trade < 0 and rebalance.action != "SELL":
            logger.warning(f"{symbol}: shares negative but action {rebalance.action}, correcting to SELL")
            rebalance.action = "SELL"
        elif rebalance.shares_to_trade > 0 and rebalance.action != "BUY":
            logger.warning(f"{symbol}: shares positive but action {rebalance.action}, correcting to BUY")
            rebalance.action = "BUY"

    def _fetch_returns_data(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch and prepare returns data for optimization.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            DataFrame with daily returns

        Raises:
            ValueError: If insufficient symbols or data
        """
        if len(symbols) < MIN_SYMBOLS:
            err_msg = f"Portfolio optimization requires at least {MIN_SYMBOLS} symbols"
            raise ValueError(err_msg)

        logger.info(f"Fetching {self.period_days} days of data for {len(symbols)} symbols")

        # Fetch price data for all symbols
        prices_data = {}
        for symbol in symbols:
            try:
                market_data = self.market_fetcher.fetch_daily(symbol, period_days=self.period_days)
                if not market_data.data.empty:
                    # Handle both lowercase and uppercase column names
                    close_col = "close" if "close" in market_data.data.columns else "Close"
                    prices_data[symbol] = market_data.data[close_col]
                else:
                    logger.warning(f"Empty data for {symbol}, skipping")
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}, skipping")

        if len(prices_data) < MIN_SYMBOLS:
            err_msg = f"Insufficient data: only {len(prices_data)}/{len(symbols)} symbols fetched"
            raise ValueError(err_msg)

        # Build aligned price DataFrame
        prices_df = pd.DataFrame(prices_data)
        prices_df = prices_df.dropna()

        if len(prices_df) < MIN_DATA_POINTS:
            err_msg = f"Insufficient data points: {len(prices_df)} < {MIN_DATA_POINTS}"
            raise ValueError(err_msg)

        if len(prices_df) < WARN_DATA_POINTS:
            logger.warning(
                f"Limited data points: {len(prices_df)} < {WARN_DATA_POINTS}, optimization may be less robust"
            )

        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        logger.info(f"Prepared returns data: {len(returns_df)} days, {len(returns_df.columns)} symbols")

        return returns_df

    def __repr__(self) -> str:
        """String representation."""
        return f"PortfolioOptimizer(period_days={self.period_days}, broker={'yes' if self.broker else 'no'})"
