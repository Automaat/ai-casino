"""Trading strategies."""

from src.strategies.ensemble import AggregationMethod, EnsembleResult, EnsembleStrategy, StrategyResult
from src.strategies.mean_reversion import MeanReversionIndicators, MeanReversionStrategy
from src.strategies.momentum import MomentumIndicators, MomentumStrategy, Signal
from src.strategies.trend_following import TrendFollowingIndicators, TrendFollowingStrategy

__all__ = [
    "AggregationMethod",
    "EnsembleResult",
    "EnsembleStrategy",
    "MeanReversionIndicators",
    "MeanReversionStrategy",
    "MomentumIndicators",
    "MomentumStrategy",
    "Signal",
    "StrategyResult",
    "TrendFollowingIndicators",
    "TrendFollowingStrategy",
]
