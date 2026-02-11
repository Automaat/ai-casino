"""Unified thesis researcher agent supporting both bullish and bearish analysis."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from src.agents.base_researcher import BaseResearcher, ResearchDirection
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.models.llm import LLMClient
from src.strategies.signal import Signal

if TYPE_CHECKING:
    from src.agents.comparative import ComparativeAnalysis
    from src.agents.trump import TrumpAnalysis


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


class ConfidenceCalculator(ABC):
    """Abstract base for direction-specific confidence calculation."""

    @abstractmethod
    def calculate(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence score.

        Args:
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result
            fundamental: Fundamental analysis result

        Returns:
            Confidence score (0.0-1.0)
        """
        ...


class BullishConfidenceCalculator(ConfidenceCalculator):
    """Confidence calculator for bullish thesis."""

    def calculate(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,  # noqa: ARG002
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence in bull case.

        Args:
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result (unused, for API consistency)
            fundamental: Fundamental analysis result

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.5  # Base confidence

        # Technical boost/penalty
        if technical.signal == Signal.BUY:
            confidence += 0.15
        elif technical.signal == Signal.SELL:
            confidence -= 0.2

        # Sentiment boost/penalty
        if sentiment.sentiment_score > 0.3:
            confidence += 0.1
        elif sentiment.sentiment_score < -0.3:
            confidence -= 0.15

        # Fundamental boost/penalty (skip if unavailable)
        if fundamental:
            if fundamental.valuation in ["UNDERVALUED", "FAIRLY_VALUED"]:
                confidence += 0.1
            elif fundamental.valuation == "OVERVALUED":
                confidence -= 0.1

            # Growth boost
            if fundamental.revenue_growth_yoy and fundamental.revenue_growth_yoy > 0.1:  # >10% growth
                confidence += 0.05

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))


class BearishConfidenceCalculator(ConfidenceCalculator):
    """Confidence calculator for bearish thesis."""

    def calculate(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,  # noqa: ARG002
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence in bear case.

        Args:
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result (unused, for API consistency)
            fundamental: Fundamental analysis result

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.5  # Base confidence

        # Technical boost/penalty (INVERTED from bullish)
        if technical.signal == Signal.SELL:
            confidence += 0.15
        elif technical.signal == Signal.BUY:
            confidence -= 0.2

        # Sentiment boost/penalty (INVERTED from bullish)
        if sentiment.sentiment_score < -0.3:
            confidence += 0.1
        elif sentiment.sentiment_score > 0.3:
            confidence -= 0.15

        # Fundamental boost/penalty (INVERTED from bullish, skip if unavailable)
        if fundamental:
            if fundamental.valuation == "OVERVALUED":
                confidence += 0.1
            elif fundamental.valuation == "UNDERVALUED":
                confidence -= 0.1

            # High debt boost (bearish signal)
            if fundamental.debt_to_equity and fundamental.debt_to_equity > 2.0:
                confidence += 0.05

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))


class ThesisResearcher(BaseResearcher):
    """Unified thesis researcher supporting both bullish and bearish analysis."""

    def __init__(self, llm_client: LLMClient, direction: ResearchDirection) -> None:
        """Initialize thesis researcher.

        Args:
            llm_client: LLM client for generating thesis
            direction: Research direction (BULLISH or BEARISH)
        """
        prompt_dir = "bullish_researcher" if direction == ResearchDirection.BULLISH else "bearish_researcher"
        super().__init__(llm_client, direction, prompt_dir)
        self._confidence_calculator = self._create_confidence_calculator()

    def _create_confidence_calculator(self) -> ConfidenceCalculator:
        """Create direction-specific confidence calculator."""
        if self.direction == ResearchDirection.BULLISH:
            return BullishConfidenceCalculator()
        return BearishConfidenceCalculator()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ThesisResearcher(direction={self.direction.value})"

    @property
    def llm_response_model(self) -> type[BaseModel]:
        """LLM response model type."""
        return ResearchLLMResponse

    async def analyze(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
        comparative: ComparativeAnalysis | None = None,
        trump_analysis: TrumpAnalysis | None = None,
    ) -> ResearchAnalysis:
        """Construct thesis from all analyses.

        Returns:
            ResearchAnalysis with thesis, key points, target, confidence
        """
        result = await super().analyze(
            symbol, technical, sentiment, news, fundamental, comparative, trump_analysis
        )
        assert isinstance(result, ResearchAnalysis)  # noqa: S101
        return result

    def _build_analysis(
        self, thesis: str, key_points: list[str], target: float | None, confidence: float
    ) -> ResearchAnalysis:
        """Build unified analysis result.

        Args:
            thesis: Thesis text
            key_points: Key strengths or weaknesses
            target: Target upside or downside percentage
            confidence: Confidence score

        Returns:
            ResearchAnalysis instance
        """
        return ResearchAnalysis(
            direction=self.direction,
            thesis=thesis,
            key_points=key_points,
            target=target,
            confidence=confidence,
        )

    def _calculate_confidence(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence using direction-specific strategy."""
        return self._confidence_calculator.calculate(technical, sentiment, news, fundamental)
