"""DI container for AI Casino application."""

from pathlib import Path

from dependency_injector import containers, providers

from src.cache.historical import HistoricalCache
from src.di.config import load_daemon_config
from src.di.providers import data as data_providers


class AppContainer(containers.DeclarativeContainer):
    """Application DI container.

    Provides config, historical cache, and 11 data fetchers + broker.
    """

    # Config path storage
    config = providers.Configuration()

    # DaemonConfig singleton - loaded via utility
    daemon_config = providers.Singleton(
        load_daemon_config,
        config_path=config.config_path,
    )

    # Historical cache singleton - shared across all fetchers
    historical_cache = providers.Singleton(
        HistoricalCache,
        db_path=None,
    )

    # Data fetchers - all Singleton
    market_fetcher = providers.Singleton(
        data_providers.create_market_fetcher,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    news_fetcher = providers.Singleton(
        data_providers.create_news_fetcher,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    fundamental_fetcher = providers.Singleton(
        data_providers.create_fundamental_fetcher,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    finnhub_fetcher = providers.Singleton(
        data_providers.create_finnhub_fetcher,
        daemon_config=daemon_config,
    )

    reddit_fetcher = providers.Singleton(
        data_providers.create_reddit_fetcher,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    truth_social_fetcher = providers.Singleton(
        data_providers.create_truth_social_fetcher,
        historical_cache=historical_cache,
    )

    stock_universe_fetcher = providers.Singleton(
        data_providers.create_stock_universe_fetcher,
    )

    websearch_fetcher = providers.Singleton(
        data_providers.create_websearch_fetcher,
    )

    earnings_fetcher = providers.Singleton(
        data_providers.create_earnings_fetcher,
    )

    comparative_fetcher = providers.Singleton(
        data_providers.create_comparative_fetcher,
    )

    alpaca_broker = providers.Singleton(
        data_providers.create_alpaca_broker,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )


def create_container(config_path: str | Path | None = None) -> AppContainer:
    """Create dependency injection container.

    Args:
        config_path: Optional path to daemon.yaml (supports ~ expansion)

    Returns:
        Configured Container instance
    """
    container = AppContainer()

    if config_path:
        # Expand ~ and resolve to absolute path
        normalized_path = Path(config_path).expanduser().resolve()
        container.config.from_dict({"config_path": normalized_path})

    return container
