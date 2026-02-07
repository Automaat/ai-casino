"""Multi-timeframe analysis models and types."""

from datetime import datetime
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, Field

from src.strategies.signal import Signal


class Timeframe(StrEnum):
    """Trading timeframe for OHLCV data."""

    DAILY = "1d"
    HOURLY = "1h"
    FIFTEEN_MIN = "15min"


class TimeframeResult(BaseModel):
    """Single timeframe analysis result."""

    timeframe: Timeframe
    signal: Signal
    rsi: float | None = None
    macd_hist: float | None = None
    interpretation: str
    confidence: float = Field(ge=0.0, le=1.0)
    indicators: dict[str, float] = Field(default_factory=dict)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TimeframeResult(timeframe={self.timeframe}, signal={self.signal}, "
            f"confidence={self.confidence:.2f})"
        )


class MultiTimeframeAnalysis(BaseModel):
    """Aggregated multi-timeframe analysis result."""

    signal: Signal
    confidence: float = Field(ge=0.0, le=1.0)
    confluence_score: float = Field(ge=0.0, le=1.0, description="Timeframe agreement (0-1)")
    timeframe_results: dict[Timeframe, TimeframeResult]
    primary_timeframe: Timeframe = Field(description="Which timeframe drove decision")
    conflict_detected: bool = Field(default=False)
    interpretation: str

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MultiTimeframeAnalysis(signal={self.signal}, confidence={self.confidence:.2f}, "
            f"confluence={self.confluence_score:.2f}, conflict={self.conflict_detected})"
        )


class MultiTimeframeData(BaseModel):
    """Container for multi-timeframe market data."""

    symbol: str
    timeframes: dict[Timeframe, pd.DataFrame]
    last_updated: datetime

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        """String representation."""
        tf_str = ", ".join(f"{tf}({len(df)} bars)" for tf, df in self.timeframes.items())
        return f"MultiTimeframeData(symbol={self.symbol}, timeframes=[{tf_str}])"
