"""Strategy selection stage I/O models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.agents.meta import StrategySelection
from src.strategies.regime import RegimeAnalysis
from src.strategies.timeframe import MultiTimeframeData, Timeframe


class StrategySelectionInput(BaseModel):
    """Input for strategy selection stage."""

    symbol: str
    market_data: pd.DataFrame | MultiTimeframeData | None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_daily_data(self) -> pd.DataFrame:
        """Extract daily timeframe data from market data.

        Returns:
            Daily OHLCV dataframe

        Raises:
            ValueError: If market data is missing or daily timeframe data is unavailable
        """
        if self.market_data is None:
            msg = "Market data is None"
            raise ValueError(msg)
        if isinstance(self.market_data, MultiTimeframeData):
            timeframes = self.market_data.timeframes
            if Timeframe.DAILY not in timeframes:
                msg = "Daily timeframe data (Timeframe.DAILY) is missing from market_data"
                raise ValueError(msg)
            return timeframes[Timeframe.DAILY]
        return self.market_data


class StrategySelectionOutput(BaseModel):
    """Output from strategy selection stage."""

    strategy_instance: Any  # Any strategy type (Momentum, Ensemble, TrendFollowing, etc.)
    strategy_name: str
    regime_analysis: RegimeAnalysis | None
    strategy_selection: StrategySelection | None

    model_config = ConfigDict(arbitrary_types_allowed=True)
