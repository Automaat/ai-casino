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
    atr_14: float | None = None
    adx: float | None = None
    exhaustion_score: float | None = None
    exhaustion_warnings: list[str] = []
    interpretation: str
    confidence: float
    ensemble_result: EnsembleResult | None = None
    multi_timeframe: MultiTimeframeAnalysis | None = None
