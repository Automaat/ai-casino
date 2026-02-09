"""Unit tests for data provider helpers."""

import pytest

from src.cache.historical import HistoricalCache
from src.daemon.config import ApiKeysConfig, DaemonConfig, TradingMode
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
from src.di.providers import data as data_providers


@pytest.fixture
def mock_daemon_config() -> DaemonConfig:
    """Create mock daemon config with test API keys."""
    return DaemonConfig(
        api_keys=ApiKeysConfig(
            alpha_vantage_api_key="test_av_key",
            marketaux_api_key="test_mx_key",
            finnhub_api_key="test_fh_key",
            alpaca_api_key="test_alpaca_key",
            alpaca_secret_key="test_alpaca_secret",
            alpaca_paper_api_key="test_paper_key",
            alpaca_paper_secret_key="test_paper_secret",
            reddit_client_id="test_reddit_id",
            reddit_client_secret="test_reddit_secret",
            reddit_user_agent="test_user_agent",
        ),
    )


@pytest.fixture
def mock_historical_cache(tmp_path) -> HistoricalCache:
    """Create temporary historical cache."""
    db_path = tmp_path / "test.db"
    return HistoricalCache(db_path=str(db_path))


def test_create_historical_cache(tmp_path, monkeypatch):
    """Test HistoricalCache creation."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache = data_providers.create_historical_cache()
    assert isinstance(cache, HistoricalCache)
    assert str(cache._db_path).startswith(str(tmp_path))


def test_create_market_fetcher(mock_daemon_config, mock_historical_cache):
    """Test MarketDataFetcher creation with config."""
    fetcher = data_providers.create_market_fetcher(mock_daemon_config, mock_historical_cache)

    assert isinstance(fetcher, MarketDataFetcher)
    assert fetcher._cache is mock_historical_cache
    assert fetcher.use_alpha_vantage is True


def test_create_market_fetcher_env_fallback(monkeypatch, mock_historical_cache):
    """Test MarketDataFetcher API key falls back to env var."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "env_key")
    config = DaemonConfig()
    fetcher = data_providers.create_market_fetcher(config, mock_historical_cache)

    assert isinstance(fetcher, MarketDataFetcher)


def test_create_news_fetcher(mock_daemon_config, mock_historical_cache):
    """Test NewsFetcher creation with config."""
    fetcher = data_providers.create_news_fetcher(mock_daemon_config, mock_historical_cache)

    assert isinstance(fetcher, NewsFetcher)
    assert fetcher._cache is mock_historical_cache
    assert fetcher.api_key == "test_mx_key"


def test_create_news_fetcher_env_fallback(monkeypatch, mock_historical_cache):
    """Test NewsFetcher API key falls back to env var."""
    monkeypatch.setenv("MARKETAUX_API_KEY", "env_key")
    config = DaemonConfig()
    fetcher = data_providers.create_news_fetcher(config, mock_historical_cache)

    assert isinstance(fetcher, NewsFetcher)


def test_create_fundamental_fetcher(mock_daemon_config, mock_historical_cache):
    """Test FundamentalDataFetcher creation with config."""
    fetcher = data_providers.create_fundamental_fetcher(mock_daemon_config, mock_historical_cache)

    assert isinstance(fetcher, FundamentalDataFetcher)
    assert fetcher._cache is mock_historical_cache
    assert fetcher.api_key == "test_av_key"


def test_create_fundamental_fetcher_env_fallback(monkeypatch, mock_historical_cache):
    """Test FundamentalDataFetcher API key falls back to env var."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "env_key")
    config = DaemonConfig()
    fetcher = data_providers.create_fundamental_fetcher(config, mock_historical_cache)

    assert isinstance(fetcher, FundamentalDataFetcher)


def test_create_finnhub_fetcher(mock_daemon_config):
    """Test FinnhubFetcher creation with config."""
    fetcher = data_providers.create_finnhub_fetcher(mock_daemon_config)

    assert isinstance(fetcher, FinnhubFetcher)
    assert fetcher._api_key == "test_fh_key"


def test_create_finnhub_fetcher_env_fallback(monkeypatch):
    """Test FinnhubFetcher API key falls back to env var."""
    monkeypatch.setenv("FINNHUB_API_KEY", "env_key")
    config = DaemonConfig()
    fetcher = data_providers.create_finnhub_fetcher(config)

    assert isinstance(fetcher, FinnhubFetcher)


def test_create_reddit_fetcher(mock_daemon_config, mock_historical_cache):
    """Test RedditFetcher creation with config."""
    fetcher = data_providers.create_reddit_fetcher(mock_daemon_config, mock_historical_cache)

    assert isinstance(fetcher, RedditFetcher)
    assert fetcher._historical_cache is mock_historical_cache


def test_create_reddit_fetcher_env_fallback(monkeypatch, mock_historical_cache):
    """Test RedditFetcher credentials fall back to env vars."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "env_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "env_secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "env_agent")
    config = DaemonConfig()
    fetcher = data_providers.create_reddit_fetcher(config, mock_historical_cache)

    assert isinstance(fetcher, RedditFetcher)


def test_create_truth_social_fetcher(mock_historical_cache):
    """Test TruthSocialFetcher creation."""
    fetcher = data_providers.create_truth_social_fetcher(mock_historical_cache)

    assert isinstance(fetcher, TruthSocialFetcher)
    assert fetcher._historical_cache is mock_historical_cache


def test_create_stock_universe_fetcher():
    """Test StockUniverseFetcher creation."""
    fetcher = data_providers.create_stock_universe_fetcher()

    assert isinstance(fetcher, StockUniverseFetcher)


def test_create_websearch_fetcher():
    """Test WebSearchFetcher creation."""
    fetcher = data_providers.create_websearch_fetcher()

    assert isinstance(fetcher, WebSearchFetcher)


def test_create_earnings_fetcher():
    """Test EarningsCalendarFetcher creation."""
    fetcher = data_providers.create_earnings_fetcher()

    assert isinstance(fetcher, EarningsCalendarFetcher)
    assert fetcher._delay == 0.5


def test_create_comparative_fetcher():
    """Test ComparativeDataFetcher creation."""
    fetcher = data_providers.create_comparative_fetcher()

    assert isinstance(fetcher, ComparativeDataFetcher)


def test_create_alpaca_broker_paper(mock_historical_cache):
    """Test AlpacaBroker creation in paper mode."""
    config = DaemonConfig(
        trading_mode=TradingMode.PAPER,
        api_keys=ApiKeysConfig(
            alpaca_paper_api_key="paper_key",
            alpaca_paper_secret_key="paper_secret",
        ),
    )
    broker = data_providers.create_alpaca_broker(config, mock_historical_cache)

    assert isinstance(broker, AlpacaBroker)
    assert broker.paper is True
    assert broker._cache is mock_historical_cache


def test_create_alpaca_broker_paper_fallback(mock_historical_cache):
    """Test AlpacaBroker paper mode falls back to regular credentials."""
    config = DaemonConfig(
        trading_mode=TradingMode.PAPER,
        api_keys=ApiKeysConfig(
            alpaca_api_key="regular_key",
            alpaca_secret_key="regular_secret",
        ),
    )
    broker = data_providers.create_alpaca_broker(config, mock_historical_cache)

    assert isinstance(broker, AlpacaBroker)
    assert broker.paper is True


def test_create_alpaca_broker_live(mock_historical_cache):
    """Test AlpacaBroker creation in live mode."""
    config = DaemonConfig(
        trading_mode=TradingMode.LIVE,
        api_keys=ApiKeysConfig(
            alpaca_api_key="live_key",
            alpaca_secret_key="live_secret",
        ),
    )
    broker = data_providers.create_alpaca_broker(config, mock_historical_cache)

    assert isinstance(broker, AlpacaBroker)
    assert broker.paper is False
    assert broker._cache is mock_historical_cache


def test_create_alpaca_broker_env_fallback(monkeypatch, mock_historical_cache):
    """Test AlpacaBroker credentials fall back to env vars."""
    monkeypatch.setenv("ALPACA_API_KEY", "env_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "env_secret")
    config = DaemonConfig()
    broker = data_providers.create_alpaca_broker(config, mock_historical_cache)

    assert isinstance(broker, AlpacaBroker)
