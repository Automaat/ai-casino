"""Bullish researcher agent for constructing optimistic investment thesis."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

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


class BullishLLMResponse(BaseModel):
    """LLM response structure for bullish research."""

    thesis: str = Field(description="Bull thesis (3-4 sentences)")
    key_strengths: list[str] = Field(description="Top 3-5 bullish signals as bullet points")
    target_upside: float | None = Field(description="Expected upside percentage or null if unavailable")


class BullishResearchAnalysis(BaseModel):
    """Bullish research analysis result."""

    thesis: str = Field(description="Bull thesis (3-4 sentences)")
    key_strengths: list[str] = Field(description="Top 3-5 bullish signals")
    target_upside: float | None = Field(description="Expected upside % or None", default=None)
    confidence: float = Field(description="Confidence in bull case (0.0-1.0)", ge=0.0, le=1.0)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BullishResearchAnalysis(thesis='{self.thesis[:50]}...', "
            f"strengths={len(self.key_strengths)}, upside={self.target_upside}, "
            f"confidence={self.confidence:.2f})"
        )


class BullishResearcher(BaseResearcher):
    """Bullish researcher agent - synthesizes optimistic case from all analyses."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize bullish researcher.

        Args:
            llm_client: LLM client for generating bull thesis
        """
        super().__init__(llm_client, ResearchDirection.BULLISH, "bullish_researcher")

    @property
    def llm_response_model(self) -> type[BaseModel]:
        """LLM response model type."""
        return BullishLLMResponse

    async def analyze(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
        comparative: "ComparativeAnalysis | None" = None,
        trump_analysis: "TrumpAnalysis | None" = None,
    ) -> BullishResearchAnalysis:
        """Construct bullish thesis from all analyses.

        Returns:
            BullishResearchAnalysis with thesis, strengths, target, confidence
        """
        result = await super().analyze(
            symbol, technical, sentiment, news, fundamental, comparative, trump_analysis
        )
        assert isinstance(result, BullishResearchAnalysis)  # noqa: S101
        return result

    def _build_analysis(
        self, thesis: str, key_points: list[str], target: float | None, confidence: float
    ) -> BullishResearchAnalysis:
        """Build bullish analysis result.

        Args:
            thesis: Bull thesis text
            key_points: Key strengths
            target: Target upside percentage
            confidence: Confidence score

        Returns:
            BullishResearchAnalysis instance
        """
        return BullishResearchAnalysis(
            thesis=thesis,
            key_strengths=key_points,
            target_upside=target,
            confidence=confidence,
        )

    def _extract_key_strengths(self, response: str) -> list[str]:
        """Extract key strengths from LLM response (wrapper for backward compatibility).

        Args:
            response: LLM response text

        Returns:
            List of strength bullet points
        """
        return self._extract_key_points(response)

    def _extract_target_upside(self, response: str) -> float | None:
        """Extract target upside from LLM response (wrapper for backward compatibility).

        Args:
            response: LLM response text

        Returns:
            Upside percentage as float or None if not available
        """
        return self._extract_target(response)

    def _calculate_confidence(
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
