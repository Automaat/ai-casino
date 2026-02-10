"""Portfolio rebalancing for daemon."""

from loguru import logger
from pydantic import BaseModel

from src.data.broker import AlpacaBroker
from src.optimization.portfolio import OptimizedPortfolio, PortfolioOptimizer, PortfolioRebalance


class RebalancingResult(BaseModel):
    """Result of portfolio rebalancing operation."""

    optimized_portfolio: OptimizedPortfolio
    rebalance_instructions: list[PortfolioRebalance]
    executed_count: int = 0
    pending_count: int = 0


class DaemonRebalancer:
    """Wrapper around PortfolioOptimizer for daemon context."""

    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        broker: AlpacaBroker | None,
        rebalance_threshold: float,
    ) -> None:
        """Initialize daemon rebalancer.

        Args:
            optimizer: Portfolio optimizer
            broker: Optional broker for auto-execution
            rebalance_threshold: Minimum weight delta to trigger rebalance
        """
        self.optimizer = optimizer
        self.broker = broker
        self.rebalance_threshold = rebalance_threshold
        logger.info(
            f"Initialized DaemonRebalancer (threshold={rebalance_threshold:.2%}, "
            f"auto_execute={'yes' if broker else 'no'})"
        )

    def run(self, watchlist: list[str], method: str, auto_execute: bool) -> RebalancingResult:
        """Run portfolio rebalancing.

        Args:
            watchlist: Watchlist symbols
            method: Optimization method (max_sharpe, min_volatility, hrp)
            auto_execute: Whether to execute rebalances via broker

        Returns:
            RebalancingResult with portfolio and execution status
        """
        logger.info(f"Running portfolio rebalancing: {method} method for {len(watchlist)} symbols")

        # Get current portfolio from broker if available
        current_portfolio = None
        universe = set(watchlist)

        if self.broker:
            try:
                current_portfolio = self.optimizer.get_current_portfolio()
                # Merge current positions with watchlist
                universe = universe.union(current_portfolio.keys())
                logger.info(
                    f"Current portfolio: {len(current_portfolio)} positions, "
                    f"combined universe: {len(universe)} symbols"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch current portfolio: {e}, using watchlist only")

        # Run optimization
        universe_list = sorted(universe)
        if method == "max_sharpe":
            optimized = self.optimizer.optimize_max_sharpe(universe_list)
        elif method == "min_volatility":
            optimized = self.optimizer.optimize_min_volatility(universe_list)
        elif method == "hrp":
            optimized = self.optimizer.optimize_hrp(universe_list)
        else:
            msg = f"Unknown optimization method: {method}"
            raise ValueError(msg)

        logger.info(
            f"Optimization complete: E[R]={optimized.expected_return:.2%}, "
            f"Vol={optimized.expected_volatility:.2%}, SR={optimized.sharpe_ratio:.2f}"
        )

        # Calculate rebalancing instructions using daemon's threshold
        rebalance_instructions = self.optimizer.calculate_rebalance(
            optimized, current_portfolio, threshold=self.rebalance_threshold
        )

        # Filter by threshold (redundant check but explicit for clarity)
        significant_rebalances = [
            r for r in rebalance_instructions if abs(r.delta) >= self.rebalance_threshold
        ]

        logger.info(
            f"Rebalancing: {len(significant_rebalances)} significant "
            f"(threshold={self.rebalance_threshold:.1%})"
        )

        # Execute if requested and broker available
        executed_count = 0
        if auto_execute and self.broker and significant_rebalances:
            executed_count = self._execute_rebalances(significant_rebalances)

        pending_count = len(significant_rebalances) - executed_count

        return RebalancingResult(
            optimized_portfolio=optimized,
            rebalance_instructions=significant_rebalances,
            executed_count=executed_count,
            pending_count=pending_count,
        )

    def _execute_rebalances(self, rebalances: list[PortfolioRebalance]) -> int:
        """Execute rebalancing orders via broker.

        Args:
            rebalances: Rebalancing instructions

        Returns:
            Number of successfully executed orders
        """
        if not self.broker:
            return 0

        executed = 0
        for rebalance in rebalances:
            if not rebalance.shares_to_trade or rebalance.action == "HOLD":
                continue

            try:
                side = "buy" if rebalance.action == "BUY" else "sell"
                shares = abs(rebalance.shares_to_trade)

                order_status = self.broker.submit_order(
                    symbol=rebalance.symbol, qty=shares, side=side, stop_loss_price=None
                )

                if order_status.filled_at is not None and order_status.filled_avg_price is not None:
                    executed += 1
                    logger.info(
                        f"Executed {side.upper()} {shares} shares of {rebalance.symbol} "
                        f"at ${order_status.filled_avg_price:.2f}"
                    )
                else:
                    logger.warning(f"Order not filled: {rebalance.symbol} {side} {shares}")

            except Exception as e:
                logger.error(f"Failed to execute {rebalance.action} {rebalance.symbol}: {e}")

        logger.info(f"Rebalancing execution: {executed}/{len(rebalances)} orders filled")
        return executed
