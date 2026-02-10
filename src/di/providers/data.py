"""Data fetcher providers for DI container."""

from src.cache.historical import HistoricalCache
from src.daemon.config import DaemonConfig
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
from src.di.config import resolve_config_or_env


def create_historical_cache() -> HistoricalCache:
    """Create HistoricalCache singleton.

    Returns:
        HistoricalCache with default db path
    """
    return HistoricalCache(db_path=None)


def create_market_fetcher(
    daemon_config: DaemonConfig,
    historical_cache: HistoricalCache,
) -> MarketDataFetcher:
    """Create MarketDataFetcher with resolved config.

    Args:
        daemon_config: Daemon configuration
        historical_cache: Shared historical cache

    Returns:
        Configured MarketDataFetcher
    """
    use_alpha_vantage = daemon_config.data_sources.market_data == "alpha_vantage"

    api_key = None
    if use_alpha_vantage:
        api_key = resolve_config_or_env(
            daemon_config.api_keys.alpha_vantage_api_key,
            "ALPHA_VANTAGE_API_KEY",
        )

    return MarketDataFetcher(
        use_alpha_vantage=use_alpha_vantage,
        api_key=api_key,
        historical_cache=historical_cache,
    )


def create_yfinance_market_fetcher(historical_cache: HistoricalCache) -> MarketDataFetcher:
    """Create yfinance-only MarketDataFetcher (no Alpha Vantage).

    Args:
        historical_cache: Shared historical cache

    Returns:
        MarketDataFetcher configured for yfinance only
    """
    return MarketDataFetcher(
        use_alpha_vantage=False,
        api_key=None,
        historical_cache=historical_cache,
    )


def create_news_fetcher(
    daemon_config: DaemonConfig,
    historical_cache: HistoricalCache,
) -> NewsFetcher:
    """Create NewsFetcher with resolved config.

    Args:
        daemon_config: Daemon configuration
        historical_cache: Shared historical cache

    Returns:
        Configured NewsFetcher
    """
    api_key = resolve_config_or_env(
        daemon_config.api_keys.marketaux_api_key,
        "MARKETAUX_API_KEY",
    )
    return NewsFetcher(
        api_key=api_key,
        historical_cache=historical_cache,
    )


def create_fundamental_fetcher(
    daemon_config: DaemonConfig,
    historical_cache: HistoricalCache,
) -> FundamentalDataFetcher:
    """Create FundamentalDataFetcher with resolved config.

    Args:
        daemon_config: Daemon configuration
        historical_cache: Shared historical cache

    Returns:
        Configured FundamentalDataFetcher
    """
    api_key = resolve_config_or_env(
        daemon_config.api_keys.alpha_vantage_api_key,
        "ALPHA_VANTAGE_API_KEY",
    )
    return FundamentalDataFetcher(
        api_key=api_key,
        historical_cache=historical_cache,
    )


def create_finnhub_fetcher(
    daemon_config: DaemonConfig,
) -> FinnhubFetcher:
    """Create FinnhubFetcher with resolved config.

    Args:
        daemon_config: Daemon configuration

    Returns:
        Configured FinnhubFetcher
    """
    api_key = resolve_config_or_env(
        daemon_config.api_keys.finnhub_api_key,
        "FINNHUB_API_KEY",
    )
    return FinnhubFetcher(
        api_key=api_key,
        cache_dir="data/cache/finnhub",
    )


def create_reddit_fetcher(
    daemon_config: DaemonConfig,
    historical_cache: HistoricalCache,
) -> RedditFetcher:
    """Create RedditFetcher with resolved config.

    Args:
        daemon_config: Daemon configuration
        historical_cache: Shared historical cache

    Returns:
        Configured RedditFetcher
    """
    client_id = resolve_config_or_env(
        daemon_config.api_keys.reddit_client_id,
        "REDDIT_CLIENT_ID",
    )
    client_secret = resolve_config_or_env(
        daemon_config.api_keys.reddit_client_secret,
        "REDDIT_CLIENT_SECRET",
    )
    user_agent = resolve_config_or_env(
        daemon_config.api_keys.reddit_user_agent,
        "REDDIT_USER_AGENT",
    )
    return RedditFetcher(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        cache_dir="data/cache/reddit",
        historical_cache=historical_cache,
    )


def create_truth_social_fetcher(
    historical_cache: HistoricalCache,
) -> TruthSocialFetcher:
    """Create TruthSocialFetcher with resolved config.

    Args:
        historical_cache: Shared historical cache

    Returns:
        Configured TruthSocialFetcher
    """
    return TruthSocialFetcher(
        cache_dir="data/cache/truth_social",
        historical_cache=historical_cache,
    )


def create_stock_universe_fetcher() -> StockUniverseFetcher:
    """Create StockUniverseFetcher.

    Returns:
        Configured StockUniverseFetcher
    """
    return StockUniverseFetcher(
        cache_dir="data/cache/universe",
    )


def create_websearch_fetcher() -> WebSearchFetcher:
    """Create WebSearchFetcher.

    Returns:
        Configured WebSearchFetcher
    """
    return WebSearchFetcher(
        cache_dir="data/cache/websearch",
    )


def create_earnings_fetcher() -> EarningsCalendarFetcher:
    """Create EarningsCalendarFetcher.

    Returns:
        Configured EarningsCalendarFetcher
    """
    return EarningsCalendarFetcher(
        delay_seconds=0.5,
    )


def create_comparative_fetcher() -> ComparativeDataFetcher:
    """Create ComparativeDataFetcher.

    Returns:
        Configured ComparativeDataFetcher
    """
    return ComparativeDataFetcher()


def create_alpaca_broker(
    daemon_config: DaemonConfig,
    historical_cache: HistoricalCache,
) -> AlpacaBroker:
    """Create AlpacaBroker with resolved config and trading mode.

    Args:
        daemon_config: Daemon configuration
        historical_cache: Shared historical cache

    Returns:
        Configured AlpacaBroker
    """
    trading_mode = daemon_config.trading_mode.value

    if trading_mode == "paper":
        paper_api_key = resolve_config_or_env(
            daemon_config.api_keys.alpaca_paper_api_key,
            "ALPACA_PAPER_API_KEY",
        )
        paper_secret_key = resolve_config_or_env(
            daemon_config.api_keys.alpaca_paper_secret_key,
            "ALPACA_PAPER_SECRET_KEY",
        )

        if paper_api_key and paper_secret_key:
            api_key = paper_api_key
            secret_key = paper_secret_key
        else:
            api_key = resolve_config_or_env(
                daemon_config.api_keys.alpaca_api_key,
                "ALPACA_API_KEY",
            )
            secret_key = resolve_config_or_env(
                daemon_config.api_keys.alpaca_secret_key,
                "ALPACA_SECRET_KEY",
            )
    else:  # live
        api_key = resolve_config_or_env(
            daemon_config.api_keys.alpaca_api_key,
            "ALPACA_API_KEY",
        )
        secret_key = resolve_config_or_env(
            daemon_config.api_keys.alpaca_secret_key,
            "ALPACA_SECRET_KEY",
        )

    return AlpacaBroker(
        api_key=api_key,
        secret_key=secret_key,
        paper=(trading_mode == "paper"),
        historical_cache=historical_cache,
    )
