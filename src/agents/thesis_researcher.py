"""Thesis research analysis models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator, model_validator

from src.agents.base_researcher import ResearchDirection
from src.strategies.signal import Signal

if TYPE_CHECKING:
    from src.agents.fundamental import FundamentalAnalysis
    from src.agents.news import NewsAnalysis
    from src.agents.sentiment import SentimentAnalysis
    from src.agents.technical import TechnicalAnalysis


class ResearchAnalysis(BaseModel):
    """Unified research analysis result."""

    direction: ResearchDirection = Field(description="Research direction (BULLISH or BEARISH)")
    thesis: str = Field(description="Investment thesis (3-4 sentences)")
    key_points: list[str] = Field(description="Key strengths or weaknesses")
    target: float | None = Field(description="Target upside/downside % or None", default=None)
    confidence: float = Field(description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_fields(cls, data: object) -> object:
        """Support backward compatibility with old field names."""
        if not isinstance(data, dict):
            return data

        data = data.copy()

        # Handle old bullish field names
        if "key_strengths" in data and "key_points" not in data:
            data["key_points"] = data.pop("key_strengths")
            if "direction" not in data:
                data["direction"] = ResearchDirection.BULLISH

        # Handle old bearish field names
        if "key_weaknesses" in data and "key_points" not in data:
            data["key_points"] = data.pop("key_weaknesses")
            if "direction" not in data:
                data["direction"] = ResearchDirection.BEARISH

        # Handle old target field names
        if "target_upside" in data and "target" not in data:
            data["target"] = data.pop("target_upside")
            if "direction" not in data:
                data["direction"] = ResearchDirection.BULLISH

        if "target_downside" in data and "target" not in data:
            data["target"] = data.pop("target_downside")
            if "direction" not in data:
                data["direction"] = ResearchDirection.BEARISH

        return data

    @property
    def key_strengths(self) -> list[str] | None:
        """Get key strengths (bullish only)."""
        return self.key_points if self.direction == ResearchDirection.BULLISH else None

    @property
    def key_weaknesses(self) -> list[str] | None:
        """Get key weaknesses (bearish only)."""
        return self.key_points if self.direction == ResearchDirection.BEARISH else None

    @property
    def target_upside(self) -> float | None:
        """Get target upside (bullish only)."""
        return self.target if self.direction == ResearchDirection.BULLISH else None

    @property
    def target_downside(self) -> float | None:
        """Get target downside (bearish only)."""
        return self.target if self.direction == ResearchDirection.BEARISH else None

    def __repr__(self) -> str:
        """String representation."""
        direction_str = "Bullish" if self.direction == ResearchDirection.BULLISH else "Bearish"
        return (
            f"{direction_str}ResearchAnalysis(thesis='{self.thesis[:50]}...', "
            f"points={len(self.key_points)}, target={self.target}, confidence={self.confidence:.2f})"
        )


# Backward compatibility type aliases
BullishResearchAnalysis = ResearchAnalysis
BearishResearchAnalysis = ResearchAnalysis


def _parse_numeric_range(text: str) -> float | None:
    """Parse a numeric value or range midpoint (e.g. "20-30" → 25.0)."""
    try:
        # Check for range: hyphen with digits on both sides (not a negative number)
        if "-" in text:
            parts = text.split("-", 1)
            lo, hi = parts[0].strip(), parts[1].strip()
            if lo and hi:
                return (float(lo) + float(hi)) / 2
        return float(text)
    except ValueError:
        return None


class ResearchLLMResponse(BaseModel):
    """Unified LLM response for thesis research."""

    thesis: str = Field(description="Investment thesis (3-4 sentences)")
    key_strengths: list[str] = Field(
        default_factory=list, description="Top 3-5 bullish signals (bullish only)"
    )
    key_weaknesses: list[str] = Field(
        default_factory=list, description="Top 3-5 bearish signals (bearish only)"
    )
    target_upside: float | None = Field(
        default=None, description="Expected upside percentage or null (bullish only)"
    )
    target_downside: float | None = Field(
        default=None, description="Expected downside percentage or null (bearish only)"
    )

    @field_validator("target_upside", "target_downside", mode="before")
    @classmethod
    def parse_target_value(cls, v: object) -> float | None:
        """Parse target percentage from various LLM output formats.

        Handles: "N/A", "20%", "20-30%" (takes midpoint), "~15", plain numbers.
        """
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if not isinstance(v, str):
            return None
        cleaned = v.strip().rstrip("%").strip("~").strip()
        if cleaned.upper() in ("N/A", "NA", "NULL", "NONE", ""):
            return None
        return _parse_numeric_range(cleaned)


class ConfidenceCalculator(ABC):
    """Abstract base for direction-specific confidence calculation."""

    @abstractmethod
    def calculate(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        _news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence score."""
        ...


class BullishConfidenceCalculator(ConfidenceCalculator):
    """Confidence calculator for bullish thesis."""

    def calculate(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        _news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence in bull case."""
        confidence = 0.5

        if technical.signal == Signal.BUY:
            confidence += 0.15
        elif technical.signal == Signal.SELL:
            confidence -= 0.2

        if sentiment.sentiment_score > 0.3:
            confidence += 0.1
        elif sentiment.sentiment_score < -0.3:
            confidence -= 0.15

        if fundamental:
            if fundamental.valuation in ["UNDERVALUED", "FAIRLY_VALUED"]:
                confidence += 0.1
            elif fundamental.valuation == "OVERVALUED":
                confidence -= 0.1
            if fundamental.revenue_growth_yoy and fundamental.revenue_growth_yoy > 0.1:
                confidence += 0.05

        return max(0.0, min(1.0, confidence))


class BearishConfidenceCalculator(ConfidenceCalculator):
    """Confidence calculator for bearish thesis."""

    def calculate(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        _news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence in bear case."""
        confidence = 0.5

        if technical.signal == Signal.SELL:
            confidence += 0.15
        elif technical.signal == Signal.BUY:
            confidence -= 0.2

        if sentiment.sentiment_score < -0.3:
            confidence += 0.1
        elif sentiment.sentiment_score > 0.3:
            confidence -= 0.15

        if fundamental:
            if fundamental.valuation == "OVERVALUED":
                confidence += 0.1
            elif fundamental.valuation == "UNDERVALUED":
                confidence -= 0.1
            if fundamental.debt_to_equity and fundamental.debt_to_equity > 2.0:
                confidence += 0.05

        return max(0.0, min(1.0, confidence))
