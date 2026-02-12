"""Portfolio-level VaR calculator using position-weighted market returns."""

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from src.data.broker import BrokerPosition
from src.data.market import MarketDataFetcher
from src.metrics.risk import RiskMetricsCalculator


class PortfolioVaRResult(BaseModel):
    """Portfolio-level VaR calculation result."""

    var_95: float = Field(description="Portfolio VaR at 95% confidence")
    var_99: float = Field(description="Portfolio VaR at 99% confidence")
    cvar_95: float = Field(description="Portfolio CVaR at 95% confidence")
    cvar_99: float = Field(description="Portfolio CVaR at 99% confidence")
    cdar_95: float = Field(description="Portfolio CDaR at 95% confidence")
    max_drawdown: float = Field(description="Portfolio max drawdown")
    portfolio_volatility: float = Field(description="Annualized portfolio volatility")
    num_positions: int = Field(description="Number of positions included")
    lookback_days: int = Field(description="Lookback period used")
    sufficient_data: bool = Field(description="Whether enough data for reliable calculation")


class PortfolioVaRCalculator:
    """Calculate portfolio-level VaR from position-weighted market returns."""

    MIN_DATA_POINTS = 2

    def __init__(
        self,
        risk_calculator: RiskMetricsCalculator,
        market_fetcher: MarketDataFetcher,
    ) -> None:
        """Initialize portfolio VaR calculator.

        Args:
            risk_calculator: Risk metrics calculator for VaR/CVaR/CDaR
            market_fetcher: Market data fetcher for position returns
        """
        self._risk_calculator = risk_calculator
        self._market_fetcher = market_fetcher
        logger.info("Initialized PortfolioVaRCalculator")

    def calculate(
        self,
        positions: dict[str, BrokerPosition],
        portfolio_value: float,
        lookback_days: int = 90,
    ) -> PortfolioVaRResult:
        """Calculate portfolio VaR from current positions.

        Args:
            positions: Current broker positions keyed by symbol
            portfolio_value: Total portfolio value
            lookback_days: Historical lookback period

        Returns:
            PortfolioVaRResult with VaR/CVaR/CDaR metrics
        """
        if not positions or portfolio_value <= 0:
            logger.warning("No positions or zero portfolio value, returning insufficient data result")
            return self._empty_result(lookback_days)

        weighted_returns, num_included = self._compute_weighted_returns(
            positions, portfolio_value, lookback_days
        )

        if len(weighted_returns) < self.MIN_DATA_POINTS:
            logger.warning(
                f"Insufficient data points ({len(weighted_returns)}), need ≥{self.MIN_DATA_POINTS}"
            )
            return self._empty_result(lookback_days)

        return self._compute_metrics(weighted_returns, num_included, lookback_days)

    def calculate_with_hypothetical(
        self,
        positions: dict[str, BrokerPosition],
        portfolio_value: float,
        new_symbol: str,
        new_position_value: float,
        lookback_days: int = 90,
    ) -> PortfolioVaRResult:
        """Calculate VaR including a hypothetical new position.

        Args:
            positions: Current broker positions
            portfolio_value: Current portfolio value
            new_symbol: Symbol of proposed new position
            new_position_value: Dollar value of proposed position
            lookback_days: Historical lookback period

        Returns:
            PortfolioVaRResult with projected VaR including new position
        """
        hypothetical_positions = dict(positions)
        new_portfolio_value = portfolio_value + new_position_value

        if new_symbol in hypothetical_positions:
            existing = hypothetical_positions[new_symbol]
            hypothetical_positions[new_symbol] = BrokerPosition(
                symbol=new_symbol,
                qty=existing.qty,
                market_value=existing.market_value + new_position_value,
                avg_entry_price=existing.avg_entry_price,
                unrealized_pnl=existing.unrealized_pnl,
                unrealized_pnl_percent=existing.unrealized_pnl_percent,
            )
        else:
            hypothetical_positions[new_symbol] = BrokerPosition(
                symbol=new_symbol,
                qty=0,
                market_value=new_position_value,
                avg_entry_price=0,
                unrealized_pnl=0,
                unrealized_pnl_percent=0,
            )

        return self.calculate(hypothetical_positions, new_portfolio_value, lookback_days)

    def _compute_weighted_returns(
        self,
        positions: dict[str, BrokerPosition],
        portfolio_value: float,
        lookback_days: int,
    ) -> tuple[list[float], int]:
        """Compute portfolio daily returns weighted by position size.

        Args:
            positions: Broker positions
            portfolio_value: Total portfolio value
            lookback_days: Lookback period

        Returns:
            Tuple of (daily portfolio returns, number of positions actually included)
        """
        symbol_returns: dict[str, pd.Series] = {}
        weights: dict[str, float] = {}

        for symbol, position in positions.items():
            weight = position.market_value / portfolio_value
            if weight <= 0:
                continue

            try:
                market_data = self._market_fetcher.fetch_daily(symbol, lookback_days)
                daily_returns = market_data.data["Close"].pct_change().dropna()
                if len(daily_returns) >= self.MIN_DATA_POINTS:
                    symbol_returns[symbol] = daily_returns
                    weights[symbol] = weight
                else:
                    logger.warning(f"Insufficient data for {symbol} ({len(daily_returns)} points), excluding")
            except Exception as e:
                logger.opt(exception=True).warning(
                    f"Failed to fetch data for {symbol}: {e}, excluding from VaR"
                )

        if not symbol_returns:
            return [], 0

        # Calculate stock weights (no normalization - preserve cash effect)
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return [], 0

        # Align all return series to common dates
        returns_df = pd.DataFrame(symbol_returns)
        returns_df = returns_df.dropna()

        if len(returns_df) < self.MIN_DATA_POINTS:
            return [], 0

        # Compute weighted portfolio returns (stock weights + cash @ 0% return)
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        for symbol, weight in weights.items():
            if symbol in returns_df.columns:
                portfolio_returns += returns_df[symbol] * weight
        # Cash contributes: cash_weight * 0.0 (implicit, documented here)

        return portfolio_returns.tolist(), len(symbol_returns)

    def _compute_metrics(
        self,
        returns: list[float],
        num_positions: int,
        lookback_days: int,
    ) -> PortfolioVaRResult:
        """Compute VaR/CVaR/CDaR from portfolio return series.

        Args:
            returns: Portfolio daily returns
            num_positions: Number of positions
            lookback_days: Lookback period used

        Returns:
            PortfolioVaRResult with all metrics
        """
        from src.metrics.risk import calculate_volatility

        var_metrics = self._risk_calculator.calculate_var(returns)
        drawdown_metrics = self._risk_calculator.calculate_cdar(returns)
        volatility = calculate_volatility(returns)

        result = PortfolioVaRResult(
            var_95=var_metrics.var_95,
            var_99=var_metrics.var_99,
            cvar_95=var_metrics.cvar_95,
            cvar_99=var_metrics.cvar_99,
            cdar_95=drawdown_metrics.cdar_95,
            max_drawdown=drawdown_metrics.max_drawdown,
            portfolio_volatility=volatility,
            num_positions=num_positions,
            lookback_days=lookback_days,
            sufficient_data=True,
        )

        logger.info(
            f"Portfolio VaR: VaR95={result.var_95:.4f}, CVaR99={result.cvar_99:.4f}, "
            f"CDaR95={result.cdar_95:.4f}, positions={num_positions}"
        )
        return result

    def _empty_result(self, lookback_days: int) -> PortfolioVaRResult:
        """Return zero-valued result for insufficient data.

        Args:
            lookback_days: Lookback period configured

        Returns:
            PortfolioVaRResult with zeros and sufficient_data=False
        """
        return PortfolioVaRResult(
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            cdar_95=0.0,
            max_drawdown=0.0,
            portfolio_volatility=0.0,
            num_positions=0,
            lookback_days=lookback_days,
            sufficient_data=False,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "PortfolioVaRCalculator()"
