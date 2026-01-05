"""Shared pytest fixtures."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.news import NewsArticle
from src.models.sentiment import SentimentScore


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing."""
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(50)],
            "High": [105 + i for i in range(50)],
            "Low": [99 + i for i in range(50)],
            "Close": [104 + i for i in range(50)],
            "Volume": [1000000] * 50,
        }
    )


@pytest.fixture
def sample_news_articles():
    """Sample news articles for testing."""
    return [
        NewsArticle(
            title="Company reports strong earnings",
            description="Quarterly earnings exceed expectations",
            url="https://example.com/1",
            published_at=datetime(2024, 1, 15, 10, 0),
            source="Reuters",
        ),
        NewsArticle(
            title="New product launch announced",
            description="Company unveils innovative product line",
            url="https://example.com/2",
            published_at=datetime(2024, 1, 15, 12, 0),
            source="Bloomberg",
        ),
        NewsArticle(
            title="Market analysts upgrade rating",
            description="Analysts raise price target",
            url="https://example.com/3",
            published_at=datetime(2024, 1, 15, 14, 0),
            source="CNBC",
        ),
    ]


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock = MagicMock()
    mock.provider = "ollama"
    mock.model = "qwen3:14b"
    mock.complete.return_value = "Mock LLM response with analysis and high confidence."
    return mock


@pytest.fixture
def mock_finbert():
    """Mock FinBERT sentiment analyzer."""
    mock = MagicMock()
    mock.device = "cpu"
    mock.analyze.return_value = SentimentScore(
        positive=0.7,
        negative=0.1,
        neutral=0.2,
    )
    mock.analyze_batch.return_value = [
        SentimentScore(positive=0.7, negative=0.1, neutral=0.2),
        SentimentScore(positive=0.6, negative=0.2, neutral=0.2),
        SentimentScore(positive=0.8, negative=0.05, neutral=0.15),
    ]
    return mock
