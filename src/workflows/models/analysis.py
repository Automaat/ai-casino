"""Analysis stage I/O models."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from src.agents.bearish_researcher import BearishResearchAnalysis
from src.agents.bullish_researcher import BullishResearchAnalysis
from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.social import SocialSentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trump import TrumpAnalysis
from src.agents.web_researcher import WebResearchAnalysis
from src.data.news import NewsArticle
from src.data.truth_social import TruthPost
from src.strategies.timeframe import MultiTimeframeData, Timeframe


class AnalysisInput(BaseModel):
    """Input for analysis stage."""

    symbol: str
    market_data: pd.DataFrame | MultiTimeframeData | None
    news_articles: list[NewsArticle] | None
    trump_posts: list[TruthPost] | None
    enable_multi_timeframe: bool

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def get_daily_data(self) -> pd.DataFrame:
        """Extract daily timeframe data from market data.

        Returns:
            Daily OHLCV dataframe

        Raises:
            ValueError: If market data is missing
        """
        if self.market_data is None:
            raise ValueError("Market data is None")
        if isinstance(self.market_data, MultiTimeframeData):
            return self.market_data.timeframes[Timeframe.DAILY]
        return self.market_data

    def get_current_price(self) -> float:
        """Extract current price from market data.

        Returns:
            Current closing price

        Raises:
            ValueError: If market data is missing or has no price data
        """
        daily_data = self.get_daily_data()
        if daily_data.empty:
            raise ValueError("Market data is empty")
        return float(daily_data["close"].iloc[-1])


class AnalysisOutput(BaseModel):
    """Output from analysis stage."""

    technical_analysis: TechnicalAnalysis | None
    sentiment_analysis: SentimentAnalysis | None
    news_analysis: NewsAnalysis | None
    trump_analysis: TrumpAnalysis | None
    fundamental_analysis: FundamentalAnalysis | None
    comparative_analysis: ComparativeAnalysis | None
    web_research: WebResearchAnalysis | None
    social_sentiment_analysis: SocialSentimentAnalysis | None
    bullish_research: BullishResearchAnalysis | None
    bearish_research: BearishResearchAnalysis | None
    warnings: list[str] = Field(default_factory=list)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
