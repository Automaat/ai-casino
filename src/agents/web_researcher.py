"""Web research analysis models."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ResearchCategory(StrEnum):
    """Categories of web research."""

    LATEST_NEWS = "latest_news"
    MARKET_SENTIMENT = "market_sentiment"
    COMPANY_INFO = "company_info"
    COMPETITOR_ANALYSIS = "competitor_analysis"


class WebResearchResult(BaseModel):
    """Result from web research for a single category."""

    category: ResearchCategory
    summary: str = Field(description="Summary of findings (2-3 sentences)")
    key_findings: list[str] = Field(description="3-5 key findings")
    sentiment_indication: str = Field(description="Bullish, Bearish, or Neutral")
    confidence: float = Field(description="Confidence in findings (0.0-1.0)", ge=0.0, le=1.0)
    sources_count: int = Field(description="Number of sources consulted", default=0)


class WebResearchAnalysis(BaseModel):
    """Complete web research analysis."""

    symbol: str
    results: list[WebResearchResult]
    overall_sentiment: str = Field(description="Aggregated sentiment: Bullish, Bearish, or Neutral")
    confidence: float = Field(description="Overall confidence (0.0-1.0)", ge=0.0, le=1.0)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WebResearchAnalysis(symbol={self.symbol}, categories={len(self.results)}, "
            f"sentiment={self.overall_sentiment}, confidence={self.confidence:.2f})"
        )


# Predefined query templates for Ollama fallback (no tool calling)
QUERY_TEMPLATES = {
    ResearchCategory.LATEST_NEWS: "{symbol} stock latest news today",
    ResearchCategory.MARKET_SENTIMENT: "{symbol} stock market sentiment analysis",
    ResearchCategory.COMPANY_INFO: "{symbol} company recent developments announcements",
    ResearchCategory.COMPETITOR_ANALYSIS: "{symbol} stock competitors comparison",
}
