"""Daemon optimization orchestrator for after-hours parameter tuning."""

from datetime import UTC, datetime, timedelta

from loguru import logger

from src.backtesting.runner import BacktestRunner
from src.optimization.optimizer import OptunaOptimizer
from src.optimization.param_store import OptimizedParamStore, SymbolStrategyParams
from src.optimization.search_space import StrategyType
from src.optimization.validation import WalkForwardValidator

DEFAULT_STRATEGIES = ["momentum", "mean_reversion", "trend_following"]


class DaemonOptimizer:
    """Orchestrates per-symbol strategy parameter optimization."""

    def __init__(
        self,
        param_store: OptimizedParamStore,
        n_trials: int = 100,
        min_trades: int = 100,
        backtest_days: int = 730,
    ) -> None:
        """Initialize daemon optimizer.

        Args:
            param_store: Store for persisting optimized params
            n_trials: Number of Optuna trials per optimization
            min_trades: Minimum trades required for valid optimization
            backtest_days: Backtest window in days (default 2 years)
        """
        self._param_store = param_store
        self._n_trials = n_trials
        self._min_trades = min_trades
        self._backtest_days = backtest_days

        validator = WalkForwardValidator(n_splits=3, train_ratio=0.7)
        self._optimizer = OptunaOptimizer(
            runner=BacktestRunner(),
            n_trials=n_trials,
            validator=validator,
        )

        logger.info(
            f"DaemonOptimizer initialized: trials={n_trials}, min_trades={min_trades}, "
            f"backtest_days={backtest_days}"
        )

    def optimize_symbol(self, symbol: str, strategy_name: str) -> SymbolStrategyParams | None:
        """Optimize parameters for a single symbol-strategy pair.

        Args:
            symbol: Stock ticker
            strategy_name: Strategy to optimize

        Returns:
            SymbolStrategyParams if optimization succeeds and meets min_trades, else None
        """
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=self._backtest_days)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        logger.info(f"Optimizing {symbol}/{strategy_name} ({start_str} to {end_str})")

        try:
            result = self._optimizer.optimize(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
                strategy_name=strategy_name,
            )
        except Exception as e:
            logger.error(f"Optimization failed for {symbol}/{strategy_name}: {e}")
            return None

        # Validate trade count via backtest with best params
        try:
            strategy_type = StrategyType(strategy_name)
            strategy_class = self._optimizer._create_strategy_class(strategy_type, result.best_params)  # noqa: SLF001
            validation_result = self._optimizer.runner.run_backtest(
                symbol=symbol, start_date=start_str, end_date=end_str, strategy_class=strategy_class
            )
            validation_trades = validation_result.total_trades
        except Exception as e:
            logger.warning(f"Validation backtest failed for {symbol}/{strategy_name}: {e}")
            validation_trades = 0

        if validation_trades < self._min_trades:
            logger.warning(
                f"Skipping {symbol}/{strategy_name}: {validation_trades} trades < {self._min_trades} minimum"
            )
            return None

        params = SymbolStrategyParams(
            symbol=symbol,
            strategy_name=strategy_name,
            params=result.best_params,
            metrics=result.best_metrics,
            optimized_at=datetime.now(UTC),
            trials_count=result.total_trials,
            validation_trades=validation_trades,
        )

        self._param_store.save(params)
        logger.info(
            f"Optimized {symbol}/{strategy_name}: sharpe={result.best_metrics.get('sharpe_ratio', 0):.2f}, "
            f"trades={validation_trades}"
        )

        return params

    def optimize_watchlist(
        self,
        watchlist: list[str],
        strategies: list[str] | None = None,
        refresh_days: int = 30,
    ) -> tuple[list[str], list[str]]:
        """Optimize all strategies for all symbols in watchlist.

        Args:
            watchlist: List of stock tickers
            strategies: Strategy names to optimize (default: all 3)
            refresh_days: Skip symbols with params newer than this

        Returns:
            Tuple of (optimized_symbols, skipped_symbols)
        """
        strategies = strategies or DEFAULT_STRATEGIES
        optimized: list[str] = []
        skipped: list[str] = []

        total = len(watchlist) * len(strategies)
        completed = 0

        for symbol in watchlist:
            symbol_optimized = False
            symbol_skipped = True

            for strategy_name in strategies:
                completed += 1

                if not self._param_store.is_stale(symbol, strategy_name, max_age_days=refresh_days):
                    logger.debug(f"Skipping {symbol}/{strategy_name}: params still fresh")
                    continue

                symbol_skipped = False
                logger.info(f"[{completed}/{total}] Optimizing {symbol}/{strategy_name}")

                result = self.optimize_symbol(symbol, strategy_name)
                if result is not None:
                    symbol_optimized = True

            if symbol_optimized:
                optimized.append(symbol)
            elif symbol_skipped:
                skipped.append(symbol)

        logger.info(f"Optimization complete: {len(optimized)} optimized, {len(skipped)} skipped")
        return optimized, skipped

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DaemonOptimizer(trials={self._n_trials}, "
            f"min_trades={self._min_trades}, backtest_days={self._backtest_days})"
        )
