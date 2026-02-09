"""Agent providers for DI container."""

from typing import TYPE_CHECKING

from src.agents.fundamental import FundamentalAnalyst
from src.agents.news import NewsAnalyst
from src.agents.sentiment import SentimentAnalyst
from src.agents.social import SocialSentimentAnalyst
from src.agents.trump import TrumpAnalyst
from src.data.finnhub import FinnhubFetcher
from src.data.fundamental import FundamentalDataFetcher
from src.data.reddit import RedditFetcher
from src.models.llm import LLMClient

if TYPE_CHECKING:
    from src.models.sentiment import FinBERTSentiment


def create_news_analyst(llm_client: LLMClient) -> NewsAnalyst:
    """Create NewsAnalyst with LLM client.

    Args:
        llm_client: LLM client for news analysis

    Returns:
        Configured NewsAnalyst
    """
    return NewsAnalyst(llm_client)


def create_sentiment_analyst(finbert_sentiment: "FinBERTSentiment") -> SentimentAnalyst:
    """Create SentimentAnalyst with FinBERT model.

    Args:
        finbert_sentiment: FinBERT sentiment analyzer

    Returns:
        Configured SentimentAnalyst
    """
    return SentimentAnalyst(finbert_sentiment)


def create_trump_analyst(llm_client: LLMClient) -> TrumpAnalyst:
    """Create TrumpAnalyst with LLM client.

    Args:
        llm_client: LLM client for Trump post analysis

    Returns:
        Configured TrumpAnalyst
    """
    return TrumpAnalyst(llm_client)


def create_fundamental_analyst(
    llm_client: LLMClient,
    fundamental_fetcher: FundamentalDataFetcher,
) -> FundamentalAnalyst:
    """Create FundamentalAnalyst with LLM client and data fetcher.

    Args:
        llm_client: LLM client for fundamental analysis
        fundamental_fetcher: Fundamental data fetcher

    Returns:
        Configured FundamentalAnalyst
    """
    return FundamentalAnalyst(llm_client, fundamental_fetcher)


def create_social_sentiment_analyst(
    llm_client: LLMClient,
    finnhub_fetcher: FinnhubFetcher,
    reddit_fetcher: RedditFetcher,
    finbert_sentiment: "FinBERTSentiment",
) -> SocialSentimentAnalyst:
    """Create SocialSentimentAnalyst with all dependencies.

    Args:
        llm_client: LLM client for interpretation
        finnhub_fetcher: Finnhub data fetcher
        reddit_fetcher: Reddit data fetcher
        finbert_sentiment: FinBERT sentiment analyzer

    Returns:
        Configured SocialSentimentAnalyst
    """
    return SocialSentimentAnalyst(llm_client, finnhub_fetcher, reddit_fetcher, finbert_sentiment)
