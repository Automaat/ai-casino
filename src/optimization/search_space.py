"""Parameter search spaces for strategy optimization."""

from enum import StrEnum

from loguru import logger
from pydantic import BaseModel


class StrategyType(StrEnum):
    """Supported strategy types for optimization."""

    MOMENTUM = "momentum"
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    ENSEMBLE = "ensemble"


class ParamRange(BaseModel):
    """Parameter range definition."""

    name: str
    low: float
    high: float
    step: float | None = None
    is_int: bool = False


class SearchSpace(BaseModel):
    """Search space for a strategy."""

    strategy: StrategyType
    params: list[ParamRange]
    constraints: list[str] | None = None

    def __repr__(self) -> str:
        """String representation."""
        return f"SearchSpace(strategy={self.strategy}, params={len(self.params)})"


MOMENTUM_SEARCH_SPACE = SearchSpace(
    strategy=StrategyType.MOMENTUM,
    params=[
        ParamRange(name="rsi_period", low=7, high=21, step=1, is_int=True),
        ParamRange(name="rsi_oversold", low=20, high=40, step=5),
        ParamRange(name="rsi_overbought", low=60, high=80, step=5),
        ParamRange(name="macd_fast", low=8, high=15, step=1, is_int=True),
        ParamRange(name="macd_slow", low=20, high=30, step=1, is_int=True),
        ParamRange(name="macd_signal", low=6, high=12, step=1, is_int=True),
    ],
    constraints=["macd_fast < macd_slow"],
)

TREND_FOLLOWING_SEARCH_SPACE = SearchSpace(
    strategy=StrategyType.TREND_FOLLOWING,
    params=[
        ParamRange(name="sma_fast", low=20, high=100, step=10, is_int=True),
        ParamRange(name="sma_slow", low=100, high=300, step=20, is_int=True),
        ParamRange(name="adx_period", low=10, high=20, step=1, is_int=True),
        ParamRange(name="adx_threshold", low=20, high=35, step=5),
    ],
    constraints=["sma_fast < sma_slow"],
)

MEAN_REVERSION_SEARCH_SPACE = SearchSpace(
    strategy=StrategyType.MEAN_REVERSION,
    params=[
        ParamRange(name="bb_period", low=10, high=30, step=5, is_int=True),
        ParamRange(name="bb_std", low=1.5, high=3.0, step=0.25),
    ],
)

ENSEMBLE_SEARCH_SPACE = SearchSpace(
    strategy=StrategyType.ENSEMBLE,
    params=[
        ParamRange(name="momentum_weight", low=0.1, high=0.6, step=0.05),
        ParamRange(name="mean_reversion_weight", low=0.1, high=0.4, step=0.05),
        ParamRange(name="trend_following_weight", low=0.1, high=0.5, step=0.05),
    ],
    constraints=["weights_normalize_to_1"],
)

SEARCH_SPACES: dict[StrategyType, SearchSpace] = {
    StrategyType.MOMENTUM: MOMENTUM_SEARCH_SPACE,
    StrategyType.TREND_FOLLOWING: TREND_FOLLOWING_SEARCH_SPACE,
    StrategyType.MEAN_REVERSION: MEAN_REVERSION_SEARCH_SPACE,
    StrategyType.ENSEMBLE: ENSEMBLE_SEARCH_SPACE,
}


def get_search_space(strategy_name: str) -> SearchSpace:
    """Get search space for strategy.

    Args:
        strategy_name: Strategy name (momentum, trend_following, mean_reversion, ensemble)

    Returns:
        SearchSpace for the strategy

    Raises:
        ValueError: If strategy not found
    """
    try:
        strategy_type = StrategyType(strategy_name.lower())
    except ValueError as e:
        valid = ", ".join(s.value for s in StrategyType)
        msg = f"Unknown strategy: {strategy_name}. Valid: {valid}"
        raise ValueError(msg) from e

    logger.debug(f"Retrieved search space for {strategy_type}")
    return SEARCH_SPACES[strategy_type]
