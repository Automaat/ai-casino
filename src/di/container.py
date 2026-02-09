"""DI container for AI Casino application."""

from pathlib import Path

from dependency_injector import containers, providers

from src.di.config import load_daemon_config
from src.di.providers import agents as agent_providers
from src.di.providers import data as data_providers
from src.di.providers import models as model_providers


class AppContainer(containers.DeclarativeContainer):
    """Application DI container.

    Provides config, a historical cache, multiple data fetchers, and a broker.
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
        data_providers.create_historical_cache,
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

    # Model providers
    llm_client = providers.Factory(
        model_providers.create_llm_client,
        daemon_config=daemon_config,
        metrics_collector=None,
    )

    finbert_sentiment = providers.Singleton(
        model_providers.create_finbert_sentiment,
        device=None,
    )

    # Agent providers
    news_analyst = providers.Factory(
        agent_providers.create_news_analyst,
        llm_client=llm_client,
    )

    sentiment_analyst = providers.Factory(
        agent_providers.create_sentiment_analyst,
        finbert_sentiment=finbert_sentiment,
    )

    trump_analyst = providers.Factory(
        agent_providers.create_trump_analyst,
        llm_client=llm_client,
    )

    fundamental_analyst = providers.Factory(
        agent_providers.create_fundamental_analyst,
        llm_client=llm_client,
        fundamental_fetcher=fundamental_fetcher,
    )

    social_sentiment_analyst = providers.Factory(
        agent_providers.create_social_sentiment_analyst,
        llm_client=llm_client,
        finnhub_fetcher=finnhub_fetcher,
        reddit_fetcher=reddit_fetcher,
        finbert_sentiment=finbert_sentiment,
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
