"""Worker providers for DI container."""

from typing import TYPE_CHECKING

from src.models.llm import LLMClient

if TYPE_CHECKING:
    from src.data.earnings import EarningsCalendarFetcher
    from src.data.fundamental import FundamentalDataFetcher
    from src.workers.fundamental import FundamentalWorker
    from src.workers.news import NewsWorker
    from src.workers.sentiment import SentimentWorker
    from src.workers.technical import TechnicalWorker


def create_technical_worker(llm_client: LLMClient) -> TechnicalWorker:
    """Create TechnicalWorker with LLM client.

    Args:
        llm_client: LLM client for generating interpretations

    Returns:
        Configured TechnicalWorker
    """
    from src.workers.technical import TechnicalWorker

    return TechnicalWorker(llm_client)


def create_sentiment_worker(finbert: object) -> SentimentWorker:
    """Create SentimentWorker with FinBERT.

    Args:
        finbert: FinBERT sentiment analyzer

    Returns:
        Configured SentimentWorker
    """
    from src.workers.sentiment import SentimentWorker

    return SentimentWorker(finbert)


def create_news_worker(llm_client: LLMClient) -> NewsWorker:
    """Create NewsWorker with LLM client.

    Args:
        llm_client: LLM client for analysis

    Returns:
        Configured NewsWorker
    """
    from src.workers.news import NewsWorker

    return NewsWorker(llm_client)


def create_fundamental_worker(
    llm_client: LLMClient,
    fundamental_fetcher: FundamentalDataFetcher,
    earnings_fetcher: EarningsCalendarFetcher,
) -> FundamentalWorker:
    """Create FundamentalWorker with LLM client and fetchers.

    Args:
        llm_client: LLM client for generating interpretations
        fundamental_fetcher: Fundamental data fetcher
        earnings_fetcher: Earnings calendar fetcher

    Returns:
        Configured FundamentalWorker
    """
    from src.workers.fundamental import FundamentalWorker

    return FundamentalWorker(llm_client, fundamental_fetcher, earnings_fetcher)
