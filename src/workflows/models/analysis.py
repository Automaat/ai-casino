"""Analysis stage I/O models."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.social import SocialSentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.thesis_researcher import BearishResearchAnalysis, BullishResearchAnalysis
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_daily_data(self) -> pd.DataFrame:
        """Extract daily timeframe data from market data.

        Returns:
            Daily OHLCV dataframe

        Raises:
            ValueError: If market data is missing or daily timeframe data is unavailable
        """
        if self.market_data is None:
            msg = "Market data is None"
            raise ValueError(msg)
        if isinstance(self.market_data, MultiTimeframeData):
            if Timeframe.DAILY not in self.market_data.timeframes:
                msg = "Daily timeframe data is missing from market data"
                raise ValueError(msg)
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
            msg = "Market data is empty"
            raise ValueError(msg)
        return float(daily_data["Close"].iloc[-1])


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

    model_config = ConfigDict(arbitrary_types_allowed=True)
