"""Test container factory and mock utilities for DI testing."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from src.daemon.config import ApiKeysConfig, DaemonConfig
from src.data.broker import BrokerAccountInfo, BrokerPosition, OrderStatus
from src.data.comparative import ComparativeData, PerformanceData
from src.data.comparative import StockInfo as ComparativeStockInfo
from src.data.market import MarketData
from src.di.container import AppContainer
from src.models.sentiment import SentimentScore


def create_test_config() -> DaemonConfig:
    """Create minimal test daemon config.

    Returns:
        DaemonConfig with test API keys
    """
    return DaemonConfig(
        api_keys=ApiKeysConfig(
            alpha_vantage_api_key="test_av_key",
            marketaux_api_key="test_marketaux_key",
        ),
    )


def create_test_container(
    config_overrides: DaemonConfig | None = None,
    temp_cache_path: Path | None = None,
    override_llm: bool = True,
    override_finbert: bool = True,
    override_fetchers: bool = True,
    override_broker: bool = False,
) -> AppContainer:
    """Create test container with common mock overrides.

    Args:
        config_overrides: Custom DaemonConfig (uses create_test_config() if None)
        temp_cache_path: Path for test cache DB (no cache override if None)
        override_llm: Override llm_client Factory with mock
        override_finbert: Override finbert_sentiment Singleton with mock
        override_fetchers: Override all data fetchers with mocks
        override_broker: Override alpaca_broker with mock

    Returns:
        AppContainer with specified overrides applied
    """
    container = AppContainer()

    # Config override
    config = config_overrides or create_test_config()
    container.daemon_config.override(config)

    # Cache override (if temp path provided)
    if temp_cache_path:
        from src.cache.historical import HistoricalCache

        test_cache = HistoricalCache(db_path=str(temp_cache_path))
        container.historical_cache.override(test_cache)

    # LLM client override (Factory pattern - override provider directly)
    if override_llm:
        from dependency_injector import providers

        mock_llm = create_mock_llm_client()
        container.llm_client.override(providers.Factory(lambda: mock_llm))

    # FinBERT override (Singleton pattern)
    if override_finbert:
        mock_finbert = create_mock_finbert()
        container.finbert_sentiment.override(mock_finbert)

    # Fetcher overrides (all Singleton pattern)
    if override_fetchers:
        container.market_fetcher.override(create_mock_market_fetcher())
        container.news_fetcher.override(create_mock_news_fetcher())
        container.fundamental_fetcher.override(create_mock_fundamental_fetcher())
        container.finnhub_fetcher.override(create_mock_finnhub_fetcher())
        container.reddit_fetcher.override(create_mock_reddit_fetcher())
        container.truth_social_fetcher.override(create_mock_truth_social_fetcher())
        container.websearch_fetcher.override(create_mock_web_search_fetcher())
        container.earnings_fetcher.override(create_mock_earnings_fetcher())
        container.comparative_fetcher.override(create_mock_comparative_fetcher())

    # Broker override (Singleton pattern)
    if override_broker:
        container.alpaca_broker.override(create_mock_alpaca_broker())

    return container


def reset_test_container(container: AppContainer, providers: list[str] | None = None) -> None:
    """Reset provider overrides for test isolation.

    Args:
        container: AppContainer to reset
        providers: List of provider names to reset (resets all if None)
    """
    all_providers = [
        "daemon_config",
        "historical_cache",
        "llm_client",
        "finbert_sentiment",
        "market_fetcher",
        "news_fetcher",
        "fundamental_fetcher",
        "finnhub_fetcher",
        "reddit_fetcher",
        "truth_social_fetcher",
        "websearch_fetcher",
        "earnings_fetcher",
        "comparative_fetcher",
        "alpaca_broker",
    ]

    providers_to_reset = providers if providers else all_providers

    for provider_name in providers_to_reset:
        if hasattr(container, provider_name):
            provider = getattr(container, provider_name)
            provider.reset_override()


# Mock creation utilities


def create_mock_llm_client() -> MagicMock:
    """Create mock LLM client for testing.

    Returns:
        Mock with acomplete/astructured configured
    """
    from src.models.providers.base import StructuredOutputError

    mock = MagicMock()
    mock.provider = "ollama"
    mock.model = "qwen3:14b"
    mock.complete.return_value = "Mock LLM response with analysis and high confidence."
    mock.acomplete = AsyncMock(return_value="Mock LLM response with analysis and high confidence.")

    # astructured raises StructuredOutputError to trigger fallback
    async def astructured_side_effect(*args: Any, **kwargs: Any) -> None:
        msg = "Mock structured output not configured"
        raise StructuredOutputError(msg, raw_response=None)

    mock.astructured = AsyncMock(side_effect=astructured_side_effect)
    mock.supports_structured_output = True
    return mock


def create_mock_finbert() -> MagicMock:
    """Create mock FinBERT sentiment analyzer.

    Returns:
        Mock with analyze/analyze_batch methods
    """
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


def create_mock_market_fetcher() -> MagicMock:
    """Create mock MarketDataFetcher returning canned OHLCV data.

    Returns:
        Mock with fetch_daily method
    """
    mock = MagicMock()

    def fetch_daily(symbol: str, period_days: int = 90) -> MarketData:
        prices = {"AAPL": (150.0, 155.0), "TSLA": (200.0, 195.0), "GOOGL": (140.0, 140.5)}
        open_price, close_price = prices.get(symbol, (100.0, 101.0))
        df = pd.DataFrame(
            {
                "Open": [open_price],
                "High": [max(open_price, close_price) + 2],
                "Low": [min(open_price, close_price) - 2],
                "Close": [close_price],
                "Volume": [1000000],
            }
        )
        return MarketData(symbol=symbol, data=df, last_updated=datetime(2024, 1, 15, 16, 0))

    mock.fetch_daily = MagicMock(side_effect=fetch_daily)
    mock.fetch_intraday = MagicMock(side_effect=fetch_daily)
    return mock


def create_mock_news_fetcher() -> MagicMock:
    """Create mock NewsDataFetcher.

    Returns:
        Mock with fetch_news method
    """
    mock = MagicMock()
    mock.api_key = "test_news_key"
    mock.fetch_news.return_value = []
    return mock


def create_mock_fundamental_fetcher() -> MagicMock:
    """Create mock FundamentalDataFetcher.

    Returns:
        Mock with fetch_overview method
    """
    mock = MagicMock()
    mock.api_key = "test_fundamental_key"
    mock.fetch_overview.return_value = {
        "Symbol": "AAPL",
        "PERatio": "28.5",
        "EPS": "6.15",
        "MarketCapitalization": "2850000000000",
    }
    return mock


def create_mock_finnhub_fetcher() -> MagicMock:
    """Create mock FinnhubDataFetcher.

    Returns:
        Mock with fetch_social_sentiment method
    """
    mock = MagicMock()
    mock.api_key = "test_finnhub_key"
    mock.fetch_social_sentiment.return_value = {"reddit": 0.5, "twitter": 0.6}
    return mock


def create_mock_reddit_fetcher() -> MagicMock:
    """Create mock RedditDataFetcher.

    Returns:
        Mock with fetch_trending_stocks method
    """
    mock = MagicMock()
    mock.fetch_trending_stocks.return_value = []
    return mock


def create_mock_truth_social_fetcher() -> MagicMock:
    """Create mock TruthSocialFetcher.

    Returns:
        Mock with fetch_recent_posts method
    """
    mock = MagicMock()
    mock.fetch_recent_posts.return_value = []
    return mock


def create_mock_web_search_fetcher() -> MagicMock:
    """Create mock WebSearchFetcher.

    Returns:
        Mock with search/search_news methods
    """
    mock = MagicMock()
    mock.search.return_value = MagicMock(results=[])
    mock.search_news.return_value = MagicMock(results=[])
    return mock


def create_mock_earnings_fetcher() -> MagicMock:
    """Create mock EarningsDataFetcher.

    Returns:
        Mock with fetch_earnings_calendar method
    """
    mock = MagicMock()
    mock.fetch_earnings_calendar.return_value = []
    return mock


def create_mock_comparative_fetcher() -> MagicMock:
    """Create mock ComparativeDataFetcher.

    Returns:
        Mock with fetch_comparative_data method
    """
    mock = MagicMock()
    mock.fetch_comparative_data.return_value = ComparativeData(
        stock_info=ComparativeStockInfo(
            symbol="AAPL",
            sector="Technology",
            industry="Consumer Electronics",
            pe_ratio=28.5,
            price_to_book=45.2,
        ),
        stock_performance=PerformanceData(ytd_return=15.0, three_month_return=8.0),
        sector_etf="XLK",
        sector_pe=32.0,
        sector_performance=PerformanceData(ytd_return=12.0, three_month_return=5.0),
        market_pe=22.0,
        market_performance=PerformanceData(ytd_return=10.0, three_month_return=4.0),
        fetched_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
    )
    return mock


def create_mock_alpaca_broker() -> MagicMock:
    """Create mock AlpacaBroker.

    Returns:
        Mock with get_account_info/submit_order methods
    """
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
