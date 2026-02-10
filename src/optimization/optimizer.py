"""Optuna-based strategy optimizer."""

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import optuna
from loguru import logger

from src.backtesting.runner import BacktestRunner
from src.backtesting.strategies import (
    EnsembleBacktestStrategy,
    MeanReversionBacktestStrategy,
    MomentumBacktestStrategy,
    TrendFollowingBacktestStrategy,
)
from src.optimization.results import OptimizationResult
from src.optimization.search_space import SearchSpace, StrategyType, get_search_space
from src.optimization.validation import WalkForwardValidator

STRATEGY_CLASSES = {
    StrategyType.MOMENTUM: MomentumBacktestStrategy,
    StrategyType.TREND_FOLLOWING: TrendFollowingBacktestStrategy,
    StrategyType.MEAN_REVERSION: MeanReversionBacktestStrategy,
    StrategyType.ENSEMBLE: EnsembleBacktestStrategy,
}

ENSEMBLE_WEIGHT_KEYS = ["momentum_weight", "mean_reversion_weight", "trend_following_weight"]


@dataclass
class _OptimizationContext:
    """Internal context for optimization objective functions."""

    symbol: str
    start_date: str
    end_date: str
    strategy_type: StrategyType
    search_space: SearchSpace


def _apply_constraint(constraint: str, params: dict[str, float | int]) -> bool:
    """Apply a single constraint, return True if constraint violated."""
    if constraint == "macd_fast < macd_slow":
        return params.get("macd_fast", 0) >= params.get("macd_slow", 0)
    if constraint == "sma_fast < sma_slow":
        return params.get("sma_fast", 0) >= params.get("sma_slow", 0)
    if constraint == "weights_normalize_to_1":
        total = sum(params.get(k, 0) for k in ENSEMBLE_WEIGHT_KEYS)
        if total <= 0:
            return True  # Constraint violated: no valid weights
        # Normalize weights in-place (intentional mutation for Optuna trial)
        for key in ENSEMBLE_WEIGHT_KEYS:
            if key in params:
                params[key] = params[key] / total
    return False


class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer for trading strategies."""

    def __init__(
        self,
        runner: BacktestRunner | None = None,
        n_trials: int = 100,
        directions: list[str] | None = None,
        validator: WalkForwardValidator | None = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            runner: BacktestRunner instance (default: new instance)
            n_trials: Number of optimization trials
            directions: Optimization directions (maximize/minimize) for objectives.
                        Default: ["maximize"] (sharpe only)
                        Multi-objective: ["maximize", "maximize", "minimize"]
                        for sharpe, return, drawdown
            validator: Walk-forward validator (optional)
        """
        self.runner = runner or BacktestRunner()
        self.n_trials = n_trials
        self.directions = directions or ["maximize"]
        self.validator = validator
        self._multi_objective = len(self.directions) > 1

        logger.info(
            f"Initialized OptunaOptimizer: trials={n_trials}, "
            f"multi_objective={self._multi_objective}, validator={validator is not None}"
        )

    def _suggest_params(self, trial: optuna.Trial, search_space: SearchSpace) -> dict[str, float | int]:
        """Suggest parameters for a trial."""
        params: dict[str, float | int] = {}

        for param in search_space.params:
            if param.is_int:
                step = int(param.step) if param.step else 1
                params[param.name] = trial.suggest_int(param.name, int(param.low), int(param.high), step=step)
            else:
                params[param.name] = trial.suggest_float(param.name, param.low, param.high, step=param.step)

        if search_space.constraints:
            for constraint in search_space.constraints:
                if _apply_constraint(constraint, params):
                    raise optuna.TrialPruned

        return params

    def _create_strategy_class(self, strategy_type: StrategyType, params: dict[str, float | int]) -> type:
        """Create strategy class with given parameters."""
        base_class = STRATEGY_CLASSES[strategy_type]

        class ConfiguredStrategy(base_class):  # type: ignore[valid-type,misc]
            pass

        for key, value in params.items():
            setattr(ConfiguredStrategy, key, value)

        return ConfiguredStrategy

    def _run_backtest_safe(
        self, symbol: str, start_date: str, end_date: str, strategy_class: type
    ) -> tuple[float, float, float] | None:
        """Run backtest and return metrics or None on failure.

        Returns None for expected failures (insufficient data, invalid params).
        Re-raises unexpected errors (bugs in strategy logic).
        """
        try:
            result = self.runner.run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                strategy_class=strategy_class,
            )
            return result.sharpe_ratio, result.total_return, abs(result.max_drawdown)
        except (ValueError, KeyError, IndexError) as e:
            # Expected: insufficient data, invalid params, missing OHLCV columns
            logger.opt(exception=True).warning(f"Backtest skipped - invalid params/data: {e}")
            return None
        except Exception as e:
            # Unexpected: strategy bugs, computation errors - should be investigated
            logger.exception("Backtest failed unexpectedly")
            raise

    def _objective(self, trial: optuna.Trial, ctx: _OptimizationContext) -> float | tuple[float, ...]:
        """Objective function for optimization."""
        params = self._suggest_params(trial, ctx.search_space)
        strategy_class = self._create_strategy_class(ctx.strategy_type, params)

        metrics = self._run_backtest_safe(ctx.symbol, ctx.start_date, ctx.end_date, strategy_class)
        if metrics is None:
            logger.warning(f"Trial {trial.number} failed")
            raise optuna.TrialPruned

        sharpe, total_return, max_drawdown = metrics
        if self._multi_objective:
            return sharpe, total_return, max_drawdown
        return sharpe

    def _objective_with_validation(
        self, trial: optuna.Trial, ctx: _OptimizationContext, start_dt: datetime, end_dt: datetime
    ) -> float | tuple[float, ...]:
        """Objective with walk-forward validation."""
        params = self._suggest_params(trial, ctx.search_space)
        strategy_class = self._create_strategy_class(ctx.strategy_type, params)

        # Train period unused: rule-based strategies don't require training
        def fold_objective(
            _train_start: datetime, _train_end: datetime, test_start: datetime, test_end: datetime
        ) -> dict[str, float]:
            metrics = self._run_backtest_safe(
                ctx.symbol, test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"), strategy_class
            )
            if metrics is None:
                return {"sharpe_ratio": 0.0, "total_return": 0.0, "max_drawdown": 1.0}
            return {"sharpe_ratio": metrics[0], "total_return": metrics[1], "max_drawdown": metrics[2]}

        validation_result = self.validator.validate(fold_objective, start_dt, end_dt)  # type: ignore[union-attr]

        sharpe = validation_result.metrics_avg.get("sharpe_ratio", 0.0)
        total_return = validation_result.metrics_avg.get("total_return", 0.0)
        max_drawdown = validation_result.metrics_avg.get("max_drawdown", 1.0)

        if self._multi_objective:
            return sharpe, total_return, max_drawdown
        return sharpe

    def _create_study(self) -> optuna.Study:
        """Create Optuna study with appropriate settings."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self._multi_objective:
            return optuna.create_study(
                directions=self.directions,
                sampler=optuna.samplers.NSGAIISampler(),
            )
        return optuna.create_study(
            direction=self.directions[0],
            sampler=optuna.samplers.TPESampler(),
        )

    def _extract_results(
        self, study: optuna.Study, ctx: _OptimizationContext
    ) -> tuple[dict[str, float | int], dict[str, float], list[dict[str, float | int]] | None]:
        """Extract best params, metrics, and pareto front from study."""
        if self._multi_objective:
            if not study.best_trials:
                msg = "No completed trials; all may have been pruned or failed"
                raise ValueError(msg)
            pareto_front = [
                {
                    **t.params,
                    "sharpe_ratio": t.values[0],
                    "total_return": t.values[1],
                    "max_drawdown": t.values[2],
                }
                for t in study.best_trials
            ]
            best_trial = max(study.best_trials, key=lambda t: t.values[0])
            best_params = best_trial.params
            best_metrics = {
                "sharpe_ratio": best_trial.values[0],
                "total_return": best_trial.values[1],
                "max_drawdown": best_trial.values[2],
            }
            return best_params, best_metrics, pareto_front

        best_params = study.best_params
        strategy_class = self._create_strategy_class(ctx.strategy_type, best_params)
        final_result = self.runner.run_backtest(
            symbol=ctx.symbol,
            start_date=ctx.start_date,
            end_date=ctx.end_date,
            strategy_class=strategy_class,
        )
        best_metrics = {
            "sharpe_ratio": final_result.sharpe_ratio,
            "total_return": final_result.total_return,
            "max_drawdown": abs(final_result.max_drawdown),
        }
        return best_params, best_metrics, None

    def optimize(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        strategy_name: str = "momentum",
    ) -> OptimizationResult:
        """Run optimization for a strategy.

        Args:
            symbol: Stock ticker
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            strategy_name: Strategy name (momentum, trend_following, mean_reversion, ensemble)

        Returns:
            OptimizationResult with best parameters and metrics
        """
        search_space = get_search_space(strategy_name)
        strategy_type = StrategyType(strategy_name.lower())

        ctx = _OptimizationContext(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy_type=strategy_type,
            search_space=search_space,
        )

        logger.info(f"Starting optimization for {symbol} with {strategy_name} strategy")

        study = self._create_study()
        start_time = time.time()

        if self.validator:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)

            study.optimize(
                lambda trial: self._objective_with_validation(trial, ctx, start_dt, end_dt),
                n_trials=self.n_trials,
                show_progress_bar=False,
            )
        else:
            study.optimize(
                lambda trial: self._objective(trial, ctx),
                n_trials=self.n_trials,
                show_progress_bar=False,
            )

        optimization_time = time.time() - start_time
        best_params, best_metrics, pareto_front = self._extract_results(study, ctx)

        result = OptimizationResult(
            strategy_name=strategy_name,
            symbol=symbol,
            best_params=best_params,
            best_metrics=best_metrics,
            pareto_front=pareto_front,
            total_trials=len(study.trials),
            optimization_time_seconds=optimization_time,
        )

        logger.info(
            f"Optimization complete: {result.total_trials} trials in {optimization_time:.1f}s, "
            f"best sharpe={best_metrics.get('sharpe_ratio', 0):.2f}"
        )

        return result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptunaOptimizer(trials={self.n_trials}, "
            f"multi_objective={self._multi_objective}, "
            f"validator={self.validator is not None})"
        )
