"""Trading strategies."""

from typing import TYPE_CHECKING

# Always import Signal (no pandas dependency)
from src.strategies.signal import Signal

# Lazy imports to avoid pandas at package init
if TYPE_CHECKING:
    from src.strategies.ensemble import AggregationMethod, EnsembleResult, EnsembleStrategy, StrategyResult
    from src.strategies.mean_reversion import MeanReversionIndicators, MeanReversionStrategy
    from src.strategies.momentum import MomentumIndicators, MomentumStrategy
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


def __getattr__(name: str) -> type:
    """Lazy import strategy classes to avoid pandas import at package init.

    Args:
        name: Attribute name to import

    Returns:
        The requested strategy class or indicator type

    Raises:
        AttributeError: If the requested attribute doesn't exist
    """
    # Ensemble imports
    if name in ("AggregationMethod", "EnsembleResult", "EnsembleStrategy", "StrategyResult"):
        from src.strategies.ensemble import (
            AggregationMethod,
            EnsembleResult,
            EnsembleStrategy,
            StrategyResult,
        )

        mapping = {
            "AggregationMethod": AggregationMethod,
            "EnsembleResult": EnsembleResult,
            "EnsembleStrategy": EnsembleStrategy,
            "StrategyResult": StrategyResult,
        }
        return mapping[name]

    # Mean reversion imports
    if name in ("MeanReversionIndicators", "MeanReversionStrategy"):
        from src.strategies.mean_reversion import MeanReversionIndicators, MeanReversionStrategy

        mapping = {
            "MeanReversionIndicators": MeanReversionIndicators,
            "MeanReversionStrategy": MeanReversionStrategy,
        }
        return mapping[name]

    # Momentum imports
    if name in ("MomentumIndicators", "MomentumStrategy"):
        from src.strategies.momentum import MomentumIndicators, MomentumStrategy

        mapping = {
            "MomentumIndicators": MomentumIndicators,
            "MomentumStrategy": MomentumStrategy,
        }
        return mapping[name]

    # Trend following imports
    if name in ("TrendFollowingIndicators", "TrendFollowingStrategy"):
        from src.strategies.trend_following import TrendFollowingIndicators, TrendFollowingStrategy

        mapping = {
            "TrendFollowingIndicators": TrendFollowingIndicators,
            "TrendFollowingStrategy": TrendFollowingStrategy,
        }
        return mapping[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
