"""Technical analysis models."""

from pydantic import BaseModel

from src.strategies.ensemble import EnsembleResult
from src.strategies.signal import Signal
from src.strategies.timeframe import MultiTimeframeAnalysis


class TechnicalAnalysis(BaseModel):
    """Technical analysis result."""

    signal: Signal
    rsi: float | None = None
    macd_hist: float | None = None
    interpretation: str
    confidence: float
    ensemble_result: EnsembleResult | None = None
    multi_timeframe: MultiTimeframeAnalysis | None = None
