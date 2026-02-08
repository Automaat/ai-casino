"""Bearish researcher agent for constructing pessimistic investment thesis."""

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


class BearishLLMResponse(BaseModel):
    """LLM response structure for bearish research."""

    thesis: str = Field(description="Bear thesis (3-4 sentences)")
    key_weaknesses: list[str] = Field(description="Top 3-5 bearish signals as bullet points")
    target_downside: float | None = Field(description="Expected downside percentage or null if unavailable")


class BearishResearchAnalysis(BaseModel):
    """Bearish research analysis result."""

    thesis: str = Field(description="Bear thesis (3-4 sentences)")
    key_weaknesses: list[str] = Field(description="Top 3-5 bearish signals")
    target_downside: float | None = Field(description="Expected downside % or None", default=None)
    confidence: float = Field(description="Confidence in bear case (0.0-1.0)", ge=0.0, le=1.0)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BearishResearchAnalysis(thesis='{self.thesis[:50]}...', "
            f"weaknesses={len(self.key_weaknesses)}, downside={self.target_downside}, "
            f"confidence={self.confidence:.2f})"
        )


class BearishResearcher(BaseResearcher):
    """Bearish researcher agent - synthesizes pessimistic case from all analyses."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize bearish researcher.

        Args:
            llm_client: LLM client for generating bear thesis
        """
        super().__init__(llm_client, ResearchDirection.BEARISH, "bearish_researcher")

    @property
    def llm_response_model(self) -> type[BaseModel]:
        """LLM response model type."""
        return BearishLLMResponse

    async def analyze(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
        comparative: "ComparativeAnalysis | None" = None,
        trump_analysis: "TrumpAnalysis | None" = None,
    ) -> BearishResearchAnalysis:
        """Construct bearish thesis from all analyses.

        Returns:
            BearishResearchAnalysis with thesis, weaknesses, target, confidence
        """
        return await super().analyze(
            symbol, technical, sentiment, news, fundamental, comparative, trump_analysis
        )

    def _build_analysis(
        self, thesis: str, key_points: list[str], target: float | None, confidence: float
    ) -> BearishResearchAnalysis:
        """Build bearish analysis result.

        Args:
            thesis: Bear thesis text
            key_points: Key weaknesses
            target: Target downside percentage
            confidence: Confidence score

        Returns:
            BearishResearchAnalysis instance
        """
        return BearishResearchAnalysis(
            thesis=thesis,
            key_weaknesses=key_points,
            target_downside=target,
            confidence=confidence,
        )

    def _extract_key_weaknesses(self, response: str) -> list[str]:
        """Extract key weaknesses from LLM response (wrapper for backward compatibility).

        Args:
            response: LLM response text

        Returns:
            List of weakness bullet points
        """
        return self._extract_key_points(response)

    def _extract_target_downside(self, response: str) -> float | None:
        """Extract target downside from LLM response (wrapper for backward compatibility).

        Args:
            response: LLM response text

        Returns:
            Downside percentage as float or None if not available
        """
        return self._extract_target(response)

    def _calculate_confidence(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        _news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence in bear case.

        Args:
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            _news: News analysis result (unused, for API consistency)
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
