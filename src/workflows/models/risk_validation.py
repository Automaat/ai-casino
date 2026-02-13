"""Models for pre-decision risk validation."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.thesis_researcher import BearishResearchAnalysis, BullishResearchAnalysis
from src.daemon.degradation import DegradationContext
from src.strategies.session import TradingSession
from src.strategies.signal import Signal
from src.strategies.timeframe import MultiTimeframeData


class RiskValidationInput(BaseModel):
    """Input for pre-decision risk validation."""

    model_config = {"arbitrary_types_allowed": True}

    symbol: str
    trading_session: TradingSession
    technical_analysis: TechnicalAnalysis | None
    sentiment_analysis: SentimentAnalysis | None
    news_analysis: NewsAnalysis | None
    fundamental_analysis: FundamentalAnalysis | None
    bullish_research: BullishResearchAnalysis | None
    bearish_research: BearishResearchAnalysis | None
    market_data: pd.DataFrame | MultiTimeframeData | None
    degradation_context: DegradationContext | None


class SignalConsistency(BaseModel):
    """Signal consistency analysis."""

    conflicting_signals: bool
    signal_distribution: dict[Signal, int] = Field(default_factory=dict)
    conflict_details: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Pre-decision validation result."""

    approved: bool
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    constraints_met: dict[str, bool] = Field(default_factory=dict)
    blocking_issues: list[str] = Field(default_factory=list)
    signal_consistency: SignalConsistency


class RiskValidationOutput(BaseModel):
    """Output from pre-decision risk validation."""

    validation_result: ValidationResult
