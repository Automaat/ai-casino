"""Monte Carlo stress testing executor for daemon integration."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.daemon.config import MonteCarloConfig
from src.daemon.state import MonteCarloRecord
from src.data.market import MarketDataFetcher
from src.metrics.monte_carlo import MonteCarloSimulator, SimulationMethod

if TYPE_CHECKING:
    from src.data.broker import AlpacaBroker


class DaemonStressTester:
    """Executes Monte Carlo stress testing on current portfolio."""

    def __init__(
        self,
        broker_client: "AlpacaBroker",
        market_fetcher: MarketDataFetcher,
        config: MonteCarloConfig,
    ) -> None:
        """Initialize stress tester.

        Args:
            broker_client: Broker client for fetching positions
            market_fetcher: Market data fetcher for historical returns
            config: Monte Carlo configuration
        """
        self.broker = broker_client
        self.market = market_fetcher
        self.config = config

    def execute(self) -> MonteCarloRecord:
        """Run Monte Carlo simulation on current portfolio.

        Returns:
            MonteCarloRecord for state persistence

        Raises:
            ValueError: If no positions or insufficient historical data
        """
        logger.info("[STRESS TEST] Fetching current positions")
        positions = self.broker.get_positions()

        if not positions:
            msg = "No positions in portfolio for stress testing"
            raise ValueError(msg)

        # Build position dict {symbol: market_value}
        position_dict = {pos.symbol: pos.market_value for pos in positions}
        total_value = sum(position_dict.values())
        symbols = list(position_dict.keys())

        logger.info(f"[STRESS TEST] Portfolio: {len(symbols)} positions, ${total_value:,.2f} total value")

        # Fetch historical returns (lookback = max(min_days, horizon * 2))
        lookback_days = max(self.config.min_historical_days, self.config.horizon_days * 2)
        logger.info(f"[STRESS TEST] Fetching {lookback_days} days of historical data")

        returns_df = self._fetch_returns(symbols, lookback_days)

        # Create simulator and run
        simulator = MonteCarloSimulator(returns_df)
        method = SimulationMethod(self.config.simulation_method)

        logger.info(
            f"[STRESS TEST] Running {self.config.num_simulations} simulations "
            f"({method.value}, {self.config.horizon_days} days)"
        )

        result = simulator.simulate(
            positions=position_dict,
            num_simulations=self.config.num_simulations,
            horizon_days=self.config.horizon_days,
            method=method,
            random_seed=self.config.random_seed,
            loss_threshold=self.config.loss_threshold,
        )

        # Check against risk tolerance
        exceeds_tolerance = result.prob_loss_gt_threshold > self.config.max_acceptable_prob
        alert_message = None

        if exceeds_tolerance:
            alert_message = (
                f"Portfolio tail risk exceeds threshold: "
                f"P(loss>{self.config.loss_threshold:.0%})={result.prob_loss_gt_threshold:.1%} "
                f"(max {self.config.max_acceptable_prob:.1%}), "
                f"Expected worst drawdown={result.expected_worst_drawdown:.1%}"
            )

        # Build record
        record = MonteCarloRecord(
            timestamp=datetime.now(UTC),
            simulation_method=result.simulation_method.value,
            num_simulations=result.num_simulations,
            horizon_days=result.horizon_days,
            prob_loss_gt_threshold=result.prob_loss_gt_threshold,
            expected_worst_drawdown=result.expected_worst_drawdown,
            var_95=result.var_95,
            cvar_95=result.cvar_95,
            median_recovery_days=result.median_recovery_days,
            exceeds_risk_tolerance=exceeds_tolerance,
            alert_message=alert_message,
            portfolio_symbols=symbols,
            total_market_value=total_value,
        )

        threshold_pct = self.config.loss_threshold
        prob_pct = result.prob_loss_gt_threshold
        logger.info(
            f"[STRESS TEST] Complete - P(loss>{threshold_pct:.0%})={prob_pct:.1%}, "
            f"VaR95={result.var_95:.1%}, CVaR95={result.cvar_95:.1%}"
        )

        return record

    def _fetch_returns(self, symbols: list[str], lookback_days: int) -> pd.DataFrame:
        """Fetch historical returns for all symbols.

        Args:
            symbols: List of stock symbols
            lookback_days: Number of days to fetch

        Returns:
            DataFrame with columns=symbols, index=dates, values=daily returns

        Raises:
            ValueError: If insufficient data for any symbol
        """
        returns_dict = {}

        for symbol in symbols:
            try:
                market_data = self.market.fetch_daily(symbol, period_days=lookback_days)
                df = market_data.data

                # Calculate daily returns first
                returns = df["close"].pct_change().dropna()

                if len(returns) < self.config.min_historical_days:
                    msg = (
                        f"{symbol}: Only {len(returns)} return days available "
                        f"(minimum {self.config.min_historical_days})"
                    )
                    raise ValueError(msg)

                returns_dict[symbol] = returns

            except Exception as e:
                logger.error(f"[STRESS TEST] Failed to fetch {symbol}: {e}")
                msg = f"Cannot fetch historical data for {symbol}: {e}"
                raise ValueError(msg) from e

        # Combine into single DataFrame (inner join to align dates)
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()  # Remove any rows with missing data

        if len(returns_df) < self.config.min_historical_days:
            msg = (
                f"After alignment, only {len(returns_df)} days available "
                f"(minimum {self.config.min_historical_days})"
            )
            raise ValueError(msg)

        logger.info(f"[STRESS TEST] Aligned returns: {len(returns_df)} days, {len(symbols)} symbols")
        return returns_df
