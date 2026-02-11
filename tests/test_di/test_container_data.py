"""Integration tests for data fetcher providers in container."""

import pytest

from src.cache.historical import HistoricalCache
from src.daemon.config import ApiKeysConfig, DaemonConfig
from src.data.broker import AlpacaBroker
from src.data.comparative import ComparativeDataFetcher
from src.data.earnings import EarningsCalendarFetcher
from src.data.finnhub import FinnhubFetcher
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.data.reddit import RedditFetcher
from src.data.truth_social import TruthSocialFetcher
from src.data.universe import StockUniverseFetcher
from src.data.websearch import WebSearchFetcher
from src.di.container import AppContainer


@pytest.fixture
def container_with_test_config(tmp_path):
    """Create container with test config."""
    container = AppContainer()

    # Override daemon_config with test values
    test_config = DaemonConfig(
        api_keys=ApiKeysConfig(
            alpha_vantage_api_key="test_av",
            marketaux_api_key="test_mx",
            finnhub_api_key="test_fh",
            alpaca_api_key="test_alpaca",
            alpaca_secret_key="test_alpaca_secret",
            reddit_client_id="test_reddit_id",
            reddit_client_secret="test_reddit_secret",
            reddit_user_agent="test_agent",
        ),
    )
    container.daemon_config.override(test_config)

    # Override historical_cache with temp db
    test_cache = HistoricalCache(db_path=str(tmp_path / "test.db"))
    container.historical_cache.override(test_cache)

    return container


def test_all_providers_callable(container_with_test_config):
    """Test all 12 data providers are callable."""
    container = container_with_test_config

    # HistoricalCache
    cache = container.historical_cache()
    assert isinstance(cache, HistoricalCache)

    # Market data fetcher
    market = container.market_fetcher()
    assert isinstance(market, MarketDataFetcher)

    # News fetcher
    news = container.news_fetcher()
    assert isinstance(news, NewsFetcher)

    # Fundamental fetcher
    fundamental = container.fundamental_fetcher()
    assert isinstance(fundamental, FundamentalDataFetcher)

    # Finnhub fetcher
    finnhub = container.finnhub_fetcher()
    assert isinstance(finnhub, FinnhubFetcher)

    # Reddit fetcher
    reddit = container.reddit_fetcher()
    assert isinstance(reddit, RedditFetcher)

    # Truth Social fetcher
    truth_social = container.truth_social_fetcher()
    assert isinstance(truth_social, TruthSocialFetcher)

    # Stock universe fetcher
    universe = container.stock_universe_fetcher()
    assert isinstance(universe, StockUniverseFetcher)

    # WebSearch fetcher
    websearch = container.websearch_fetcher()
    assert isinstance(websearch, WebSearchFetcher)

    # Earnings calendar fetcher
    earnings = container.earnings_fetcher()
    assert isinstance(earnings, EarningsCalendarFetcher)

    # Comparative data fetcher
    comparative = container.comparative_fetcher()
    assert isinstance(comparative, ComparativeDataFetcher)

    # Alpaca broker
    broker = container.alpaca_broker()
    assert isinstance(broker, AlpacaBroker)


def test_singleton_behavior(container_with_test_config):
    """Test providers return same instance on multiple calls."""
    container = container_with_test_config

    # HistoricalCache
    cache1 = container.historical_cache()
    cache2 = container.historical_cache()
    assert cache1 is cache2

    # MarketDataFetcher
    market1 = container.market_fetcher()
    market2 = container.market_fetcher()
    assert market1 is market2

    # NewsFetcher
    news1 = container.news_fetcher()
    news2 = container.news_fetcher()
    assert news1 is news2

    # FundamentalDataFetcher
    fundamental1 = container.fundamental_fetcher()
    fundamental2 = container.fundamental_fetcher()
    assert fundamental1 is fundamental2

    # FinnhubFetcher
    finnhub1 = container.finnhub_fetcher()
    finnhub2 = container.finnhub_fetcher()
    assert finnhub1 is finnhub2

    # RedditFetcher
    reddit1 = container.reddit_fetcher()
    reddit2 = container.reddit_fetcher()
    assert reddit1 is reddit2

    # TruthSocialFetcher
    truth1 = container.truth_social_fetcher()
    truth2 = container.truth_social_fetcher()
    assert truth1 is truth2

    # StockUniverseFetcher
    universe1 = container.stock_universe_fetcher()
    universe2 = container.stock_universe_fetcher()
    assert universe1 is universe2

    # WebSearchFetcher
    websearch1 = container.websearch_fetcher()
    websearch2 = container.websearch_fetcher()
    assert websearch1 is websearch2

    # EarningsCalendarFetcher
    earnings1 = container.earnings_fetcher()
    earnings2 = container.earnings_fetcher()
    assert earnings1 is earnings2

    # ComparativeDataFetcher
    comparative1 = container.comparative_fetcher()
    comparative2 = container.comparative_fetcher()
    assert comparative1 is comparative2

    # AlpacaBroker
    broker1 = container.alpaca_broker()
    broker2 = container.alpaca_broker()
    assert broker1 is broker2


def test_shared_historical_cache(container_with_test_config):
    """Test all fetchers share same HistoricalCache instance."""
    container = container_with_test_config

    cache = container.historical_cache()
    market = container.market_fetcher()
    news = container.news_fetcher()
    fundamental = container.fundamental_fetcher()
    reddit = container.reddit_fetcher()
    truth_social = container.truth_social_fetcher()
    broker = container.alpaca_broker()

    # All fetchers with historical_cache should reference same instance
    assert market._cache is cache
    assert news._cache is cache
    assert fundamental._cache is cache
    assert reddit._historical_cache is cache
    assert truth_social._historical_cache is cache
    assert broker._cache is cache


def test_override_capability(tmp_path):
    """Test providers can be overridden for testing."""
    container = AppContainer()

    # Create mock cache
    mock_cache = HistoricalCache(db_path=str(tmp_path / "mock.db"))

    # Override historical_cache
    container.historical_cache.override(mock_cache)

    # Verify override
    cache = container.historical_cache()
    assert cache is mock_cache

    # Reset override
    container.historical_cache.reset_override()


def test_no_config_defaults(tmp_path):
    """Test container works with default DaemonConfig (env vars only)."""
    container = AppContainer()

    # Override just the cache (to avoid ~/.ai-casino)
    test_cache = HistoricalCache(db_path=str(tmp_path / "test.db"))
    container.historical_cache.override(test_cache)

    # Should not raise - uses default DaemonConfig
    config = container.daemon_config()
    assert config is not None

    # Providers should be callable (even if API keys missing)
    # Market fetcher will raise on instantiation if key missing
    from contextlib import suppress

    with suppress(ValueError):
        container.market_fetcher()

    # Fetchers without required keys should instantiate
    universe = container.stock_universe_fetcher()
    assert isinstance(universe, StockUniverseFetcher)

    websearch = container.websearch_fetcher()
    assert isinstance(websearch, WebSearchFetcher)

    earnings = container.earnings_fetcher()
    assert isinstance(earnings, EarningsCalendarFetcher)

    comparative = container.comparative_fetcher()
    assert isinstance(comparative, ComparativeDataFetcher)


def test_api_key_resolution_priority(tmp_path, monkeypatch):
    """Test config values take priority over env vars."""
    # Set env vars
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "env_key")
    monkeypatch.setenv("MARKETAUX_API_KEY", "env_mx_key")

    # Create container with config values
    container = AppContainer()
    test_config = DaemonConfig(
        api_keys=ApiKeysConfig(
            alpha_vantage_api_key="config_key",
            marketaux_api_key="config_mx_key",
        ),
    )
    container.daemon_config.override(test_config)

    test_cache = HistoricalCache(db_path=str(tmp_path / "test.db"))
    container.historical_cache.override(test_cache)

    # Verify config values used (not env)
    # Market fetcher instantiation confirms config resolved
    _ = container.market_fetcher()

    news = container.news_fetcher()
    assert news.api_key == "config_mx_key"
