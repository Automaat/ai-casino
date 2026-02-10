"""Data fetch stage output model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.data.news import NewsArticle
from src.data.truth_social import TruthPost
from src.strategies.session import TradingSession
from src.strategies.timeframe import MultiTimeframeData

if TYPE_CHECKING:
    import pandas as pd


class FetchDataOutput(BaseModel):
    """Output from data fetch stage."""

    symbol: str
    trading_session: TradingSession
    market_data: pd.DataFrame | MultiTimeframeData | None
    news_articles: list[NewsArticle] | None
    trump_posts: list[TruthPost] | None
    enable_multi_timeframe: bool
    warnings: list[str] = Field(default_factory=list)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
