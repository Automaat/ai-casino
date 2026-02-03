"""Optimization result models."""

from pydantic import BaseModel


class TrialResult(BaseModel):
    """Single trial result."""

    trial_number: int
    params: dict[str, float | int]
    metrics: dict[str, float]


class OptimizationResult(BaseModel):
    """Optuna optimization result."""

    strategy_name: str
    symbol: str
    best_params: dict[str, float | int]
    best_metrics: dict[str, float]
    pareto_front: list[dict[str, float | int]] | None = None
    total_trials: int
    optimization_time_seconds: float

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptimizationResult(strategy={self.strategy_name}, symbol={self.symbol}, "
            f"trials={self.total_trials}, sharpe={self.best_metrics.get('sharpe_ratio', 0):.2f})"
        )
