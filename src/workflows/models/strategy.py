"""Strategy selection stage I/O models."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from src.agents.meta import StrategySelection
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.regime import RegimeAnalysis
from src.strategies.timeframe import MultiTimeframeData, Timeframe


class StrategySelectionInput(BaseModel):
    """Input for strategy selection stage."""

    symbol: str
    market_data: pd.DataFrame | MultiTimeframeData | None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def get_daily_data(self) -> pd.DataFrame:
        """Extract daily timeframe data from market data.

        Returns:
            Daily OHLCV dataframe

        Raises:
            ValueError: If market data is missing
        """
        if self.market_data is None:
            raise ValueError("Market data is None")
        if isinstance(self.market_data, MultiTimeframeData):
            return self.market_data.timeframes[Timeframe.DAILY]
        return self.market_data


class StrategySelectionOutput(BaseModel):
    """Output from strategy selection stage."""

    strategy_instance: MomentumStrategy | EnsembleStrategy
    strategy_name: str
    regime_analysis: RegimeAnalysis | None
    strategy_selection: StrategySelection | None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
