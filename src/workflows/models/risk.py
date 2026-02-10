"""Risk assessment stage I/O models."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from src.agents.risk import AccountInfo, RiskAssessment
from src.daemon.degradation import DegradationContext
from src.agents.trader import TradingDecision
from src.data.broker import BrokerPosition
from src.strategies.timeframe import MultiTimeframeData, Timeframe
from src.workflows.types import BacktestValidation


class RiskAssessmentInput(BaseModel):
    """Input for risk assessment stage."""

    symbol: str
    market_data: pd.DataFrame | MultiTimeframeData | None
    final_decision: TradingDecision
    account_info: AccountInfo | None
    broker_positions: dict[str, BrokerPosition] | None
    portfolio_value: float | None
    target_portfolio_weight: float | None
    backtest_validation: BacktestValidation | None
    degradation_context: DegradationContext | None
    broker_api_failed: bool

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

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
                msg = "Daily timeframe data is missing from market data"
                raise ValueError(msg)
            return timeframes[Timeframe.DAILY]
        return self.market_data

    def get_current_price(self) -> float:
        """Extract current price from market data.

        Returns:
            Current closing price

        Raises:
            ValueError: If market data is missing or has no price data
        """
        daily_data = self.get_daily_data()
        if daily_data.empty:
            msg = "Market data is empty"
            raise ValueError(msg)
        return float(daily_data["Close"].iloc[-1])


class RiskAssessmentOutput(BaseModel):
    """Output from risk assessment stage."""

    risk_assessment: RiskAssessment

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
