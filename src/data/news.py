"""News data fetcher for financial news."""

import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

load_dotenv()


class NewsArticle(BaseModel):
    """Single news article."""

    title: str
    description: str
    url: str
    published_at: datetime
    source: str


class NewsFetcher:
    """Fetch financial news from Marketaux API."""

    BASE_URL = "https://api.marketaux.com/v1/news/all"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize news fetcher.

        Args:
            api_key: Marketaux API key. Defaults to env variable.
        """
        self.api_key = api_key or os.getenv("MARKETAUX_API_KEY", "")
        if not self.api_key:
            logger.warning("MARKETAUX_API_KEY not set - API calls may be limited")

    def fetch_company_news(
        self,
        symbol: str,
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Fetch recent news for a company.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"Fetching {limit} news articles for {symbol}")

        params = {
            "symbols": symbol,
            "filter_entities": "true",
            "limit": limit,
            "language": "en",
        }

        if self.api_key:
            params["api_token"] = self.api_key

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            articles = []

            for item in data.get("data", []):
                articles.append(
                    NewsArticle(
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        url=item.get("url", ""),
                        published_at=datetime.fromisoformat(
                            item.get("published_at", "").replace("Z", "+00:00")
                        ),
                        source=item.get("source", ""),
                    )
                )

            logger.info(f"Fetched {len(articles)} articles")
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"News fetch failed: {e}")
            raise

    def fetch_market_news(self, limit: int = 20) -> list[NewsArticle]:
        """Fetch general market news.

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"Fetching {limit} general market news articles")

        params = {
            "filter_entities": "true",
            "limit": limit,
            "language": "en",
        }

        if self.api_key:
            params["api_token"] = self.api_key

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            articles = []

            for item in data.get("data", []):
                articles.append(
                    NewsArticle(
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        url=item.get("url", ""),
                        published_at=datetime.fromisoformat(
                            item.get("published_at", "").replace("Z", "+00:00")
                        ),
                        source=item.get("source", ""),
                    )
                )

            logger.info(f"Fetched {len(articles)} articles")
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Market news fetch failed: {e}")
            raise

    def __repr__(self) -> str:
        """String representation."""
        has_key = bool(self.api_key)
        return f"NewsFetcher(authenticated={has_key})"
