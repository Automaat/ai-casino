"""Shared pytest fixtures."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.agents.bullish_researcher import BullishResearchAnalysis
from src.agents.risk import AccountInfo
from src.data.broker import BrokerAccountInfo, BrokerPosition, OrderStatus
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


@pytest.fixture
def account_info():
    """Sample account info for risk testing."""
    return AccountInfo(
        balance=100000.0,
        available_cash=80000.0,
        positions={"SPY": 100.0},
        total_exposure=20000.0,
    )


@pytest.fixture
def mock_alpaca_broker():
    """Mock Alpaca broker for testing."""
    mock = MagicMock()
    mock.paper = True

    mock.get_account_info.return_value = BrokerAccountInfo(
        balance=100000.0,
        available_cash=80000.0,
        positions={
            "AAPL": BrokerPosition(
                symbol="AAPL",
                qty=10.0,
                market_value=1500.0,
                avg_entry_price=150.0,
                unrealized_pnl=50.0,
                unrealized_pnl_percent=0.033,
            )
        },
        total_exposure=1500.0,
        portfolio_value=100000.0,
    )

    mock.submit_order.return_value = OrderStatus(
        order_id="order-123",
        symbol="AAPL",
        qty=10.0,
        filled_qty=10.0,
        side="buy",
        status="filled",
        submitted_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
        filled_at=datetime(2024, 1, 1, 10, 0, 5, tzinfo=UTC),
        filled_avg_price=150.0,
    )

    return mock


@pytest.fixture
def sample_fundamental_overview():
    """Sample fundamental overview data from Alpha Vantage."""
    return {
        "Symbol": "AAPL",
        "AssetType": "Common Stock",
        "Name": "Apple Inc",
        "PERatio": "28.5",
        "EPS": "6.15",
        "QuarterlyRevenueGrowthYOY": "0.062",
        "QuarterlyEarningsGrowthYOY": "0.102",
        "DebtToEquity": "2.05",
        "CurrentRatio": "0.94",
        "MarketCapitalization": "2850000000000",
        "EBITDA": "125000000000",
        "PriceToBookRatio": "45.2",
        "DividendYield": "0.0052",
    }


@pytest.fixture
def mock_fundamental_fetcher(sample_fundamental_overview):
    """Mock fundamental data fetcher."""
    mock = MagicMock()
    mock.api_key = "test_api_key"
    mock.fetch_overview.return_value = sample_fundamental_overview
    return mock


@pytest.fixture
def sample_bullish_research():
    """Sample bullish research analysis for testing."""
    return BullishResearchAnalysis(
        thesis=(
            "Stock shows strong momentum with improving fundamentals and positive market sentiment. "
            "Technical indicators suggest continued upward trajectory. Undervalued at current levels."
        ),
        key_strengths=[
            "Strong technical momentum with positive RSI and MACD",
            "Positive sentiment across recent news articles",
            "Undervalued relative to growth potential",
            "Strong revenue growth trajectory",
        ],
        target_upside=25.0,
        confidence=0.8,
    )
