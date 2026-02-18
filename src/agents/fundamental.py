"""Fundamental analysis models."""

from pydantic import BaseModel

from src.agents.models import EarningsFlags


class FundamentalAnalysis(BaseModel):
    """Fundamental analysis result."""

    valuation: str  # UNDERVALUED | FAIRLY_VALUED | OVERVALUED
    earnings_flags: EarningsFlags | None = None
    pe_ratio: float | None
    eps: float | None
    revenue_growth_yoy: float | None
    earnings_growth_yoy: float | None
    debt_to_equity: float | None
    current_ratio: float | None
    interpretation: str
    confidence: float
