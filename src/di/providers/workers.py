"""Worker provider functions for DI container."""

from typing import Literal

from src.data.comparative import ComparativeDataFetcher
from src.data.earnings import EarningsCalendarFetcher
from src.data.finnhub import FinnhubFetcher
from src.data.fundamental import FundamentalDataFetcher
from src.data.reddit import RedditFetcher
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.tools.websearch import WebSearchTool
from src.workers.comparative import ComparativeWorker
from src.workers.fundamental import FundamentalWorker
from src.workers.news import NewsWorker
from src.workers.sentiment import SentimentWorker
from src.workers.social import SocialSentimentWorker
from src.workers.technical import TechnicalWorker
from src.workers.thesis_research import ThesisResearchWorker
from src.workers.trump import TrumpWorker
from src.workers.web_research import WebResearchWorker


def create_technical_worker(llm_client: LLMClient) -> TechnicalWorker:
    """Create TechnicalWorker.

    Args:
        llm_client: LLM client

    Returns:
        TechnicalWorker instance
    """
    return TechnicalWorker(llm_client)


def create_sentiment_worker(finbert: FinBERTSentiment) -> SentimentWorker:
    """Create SentimentWorker.

    Args:
        finbert: FinBERT sentiment model

    Returns:
        SentimentWorker instance
    """
    return SentimentWorker(finbert)


def create_news_worker(llm_client: LLMClient) -> NewsWorker:
    """Create NewsWorker.

    Args:
        llm_client: LLM client

    Returns:
        NewsWorker instance
    """
    return NewsWorker(llm_client)


def create_fundamental_worker(
    llm_client: LLMClient,
    fundamental_fetcher: FundamentalDataFetcher,
    earnings_fetcher: EarningsCalendarFetcher,
) -> FundamentalWorker:
    """Create FundamentalWorker.

    Args:
        llm_client: LLM client
        fundamental_fetcher: Fundamental data fetcher
        earnings_fetcher: Earnings calendar fetcher

    Returns:
        FundamentalWorker instance
    """
    return FundamentalWorker(llm_client, fundamental_fetcher, earnings_fetcher)


def create_comparative_worker(
    llm_client: LLMClient, comparative_fetcher: ComparativeDataFetcher
) -> ComparativeWorker:
    """Create ComparativeWorker.

    Args:
        llm_client: LLM client
        comparative_fetcher: Comparative data fetcher

    Returns:
        ComparativeWorker instance
    """
    return ComparativeWorker(llm_client, comparative_fetcher)


def create_web_research_worker(llm_client: LLMClient, search_tool: WebSearchTool) -> WebResearchWorker:
    """Create WebResearchWorker.

    Args:
        llm_client: LLM client
        search_tool: Web search tool for data fetching

    Returns:
        WebResearchWorker instance
    """
    return WebResearchWorker(llm_client, search_tool)


def create_thesis_worker(
    llm_client: LLMClient, direction: Literal["bullish", "bearish"]
) -> ThesisResearchWorker:
    """Create ThesisResearchWorker.

    Args:
        llm_client: LLM client
        direction: Research direction (bullish or bearish)

    Returns:
        ThesisResearchWorker instance
    """
    return ThesisResearchWorker(llm_client, direction)


def create_trump_worker(llm_client: LLMClient) -> TrumpWorker:
    """Create TrumpWorker.

    Args:
        llm_client: LLM client

    Returns:
        TrumpWorker instance
    """
    return TrumpWorker(llm_client)


def create_social_sentiment_worker(
    llm_client: LLMClient,
    finnhub_fetcher: FinnhubFetcher,
    reddit_fetcher: RedditFetcher,
    finbert: FinBERTSentiment,
) -> SocialSentimentWorker:
    """Create SocialSentimentWorker.

    Args:
        llm_client: LLM client
        finnhub_fetcher: Finnhub data fetcher
        reddit_fetcher: Reddit data fetcher
        finbert: FinBERT sentiment model

    Returns:
        SocialSentimentWorker instance
    """
    return SocialSentimentWorker(llm_client, finnhub_fetcher, reddit_fetcher, finbert)
