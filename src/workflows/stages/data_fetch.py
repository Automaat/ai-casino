"""Data fetch stage implementation."""

from __future__ import annotations

import asyncio
import zoneinfo
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.data.broker import AlpacaBroker
    from src.data.market import MarketDataFetcher
    from src.data.news import NewsFetcher
    from src.data.truth_social import TruthSocialFetcher

from src.agents.risk import AccountInfo
from src.data.broker import BrokerAccountInfo, BrokerAPIError
from src.data.market import MarketData
from src.data.truth_social import TrumpPostData, TruthPost
from src.strategies.session import TradingSession
from src.strategies.timeframe import MultiTimeframeData, Timeframe
from src.workflows.models.account import AccountInfoOutput
from src.workflows.models.data_fetch import FetchDataOutput

ET_TIMEZONE = zoneinfo.ZoneInfo("America/New_York")
MARKET_HOURS_START = 4
MARKET_HOURS_END = 20


@dataclass
class DataFetchConfig:
    """Configuration for data fetching."""

    market_fetcher: MarketDataFetcher
    news_fetcher: NewsFetcher
    enable_multi_timeframe: bool = False
    trump_mode: bool = False
    trump_fetcher: TruthSocialFetcher | None = None


def _is_market_hours() -> bool:
    """Check if currently within market hours (4am-8pm ET)."""
    now = datetime.now(ET_TIMEZONE)
    return MARKET_HOURS_START <= now.hour < MARKET_HOURS_END


async def _fetch_all_data(
    symbol: str,
    period_days: int,
    use_multi_timeframe: bool,
    config: DataFetchConfig,
) -> tuple[MarketData | MultiTimeframeData, list, TrumpPostData | None]:
    """Fetch market, news, and Trump data in parallel.

    Args:
        symbol: Stock ticker
        period_days: Historical data period
        use_multi_timeframe: Whether to actually use multi-timeframe (checked at market hours)
        config: Data fetcher configuration

    Returns:
        Tuple of (market_data, news_articles, trump_data)
    """

    async def fetch_market() -> MarketData | MultiTimeframeData:
        if use_multi_timeframe:
            logger.info("Multi-timeframe mode enabled (market hours)")
            return await config.market_fetcher.fetch_multi_timeframe(
                symbol, [Timeframe.DAILY, Timeframe.HOURLY], period_days
            )
        if config.enable_multi_timeframe:
            logger.info("Multi-timeframe requested but outside market hours, using daily only")
        return await asyncio.to_thread(config.market_fetcher.fetch_daily, symbol, period_days)

    async def fetch_news_safe() -> list:
        try:
            return await asyncio.to_thread(config.news_fetcher.fetch_company_news, symbol, limit=10)
        except Exception as e:
            logger.warning(f"News fetch failed, continuing with empty news: {e}")
            return []

    async def fetch_trump_safe() -> TrumpPostData | None:
        if not config.trump_fetcher:
            return None
        try:
            return await asyncio.to_thread(config.trump_fetcher.fetch_recent, hours=24)
        except Exception as e:
            logger.warning(f"Failed to fetch Trump posts: {e}")
            return None

    async with asyncio.TaskGroup() as tg:
        market_task = tg.create_task(fetch_market())
        news_task = tg.create_task(fetch_news_safe())
        trump_task = (
            tg.create_task(fetch_trump_safe()) if config.trump_mode and config.trump_fetcher else None
        )

    news_result = news_task.result()
    # fetch_news_safe always returns list, no validation needed
    trump_data = trump_task.result() if trump_task else None

    return market_task.result(), news_result, trump_data


def _process_fetch_results(
    market_result: MarketData | MultiTimeframeData,
    trump_data: TrumpPostData | None,
    use_multi_timeframe: bool,
) -> tuple[pd.DataFrame | MultiTimeframeData, list[TruthPost] | None]:
    """Process fetched data and extract relevant fields."""
    if use_multi_timeframe:
        if not isinstance(market_result, MultiTimeframeData):
            msg = f"Expected MultiTimeframeData, got {type(market_result).__name__}"
            raise TypeError(msg)
        market_data: pd.DataFrame | MultiTimeframeData = market_result
    else:
        if not isinstance(market_result, MarketData):
            msg = f"Expected MarketData, got {type(market_result).__name__}"
            raise TypeError(msg)
        market_data = market_result.data

    trump_posts = None
    if trump_data:
        if not isinstance(trump_data, TrumpPostData):
            msg = f"Expected TrumpPostData, got {type(trump_data).__name__}"
            raise TypeError(msg)
        trump_posts = trump_data.posts
        logger.info(f"Fetched {len(trump_posts)} Trump posts")

    return market_data, trump_posts


async def fetch_data(
    symbol: str,
    period_days: int,
    trading_session: TradingSession,
    market_fetcher: MarketDataFetcher,
    news_fetcher: NewsFetcher,
    enable_multi_timeframe: bool = False,
    trump_mode: bool = False,
    trump_fetcher: TruthSocialFetcher | None = None,
    config: DataFetchConfig | None = None,
) -> FetchDataOutput:
    """Fetch market and news data (async, parallel execution).

    Args:
        symbol: Stock ticker
        period_days: Historical data period
        trading_session: Trading session type
        market_fetcher: Market data fetcher
        news_fetcher: News data fetcher
        enable_multi_timeframe: Enable multi-timeframe data fetching
        trump_mode: Enable Trump social media analysis
        trump_fetcher: Trump social media fetcher (required if trump_mode=True)
        config: Data fetch configuration (optional, overrides individual params)

    Returns:
        FetchDataOutput with market and news data
    """
    # Use config if provided, otherwise construct from individual params
    if config is None:
        config = DataFetchConfig(
            market_fetcher=market_fetcher,
            news_fetcher=news_fetcher,
            enable_multi_timeframe=enable_multi_timeframe,
            trump_mode=trump_mode,
            trump_fetcher=trump_fetcher,
        )

    logger.info("Fetching market and news data")
    use_multi_timeframe = config.enable_multi_timeframe and _is_market_hours()

    market_result, news_result, trump_data = await _fetch_all_data(
        symbol,
        period_days,
        use_multi_timeframe,
        config,
    )

    market_data, trump_posts = _process_fetch_results(market_result, trump_data, use_multi_timeframe)

    return FetchDataOutput(
        symbol=symbol,
        trading_session=trading_session,
        market_data=market_data,
        news_articles=news_result,
        trump_posts=trump_posts,
        enable_multi_timeframe=config.enable_multi_timeframe,
        warnings=[],
    )


async def _get_account_info_internal(
    broker: AlpacaBroker | None,
) -> tuple[AccountInfo, BrokerAccountInfo | None, bool]:
    """Get account information (async, thread-offloaded).

    Args:
        broker: Optional Alpaca broker

    Returns:
        Tuple of (AccountInfo, BrokerAccountInfo | None, account_info_valid: bool)
    """
    # Safe case: intentional paper trading
    if not broker:
        return (
            AccountInfo(
                balance=100000.0,
                available_cash=100000.0,
                positions={},
                total_exposure=0.0,
            ),
            None,
            True,
        )  # No broker = safe mock mode

    # Dangerous case: broker configured but API fails
    def _sync_get_account() -> tuple[AccountInfo, BrokerAccountInfo | None, bool]:
        try:
            # Broker already checked for None above
            broker_info = broker.get_account_info()  # type: ignore[union-attr]
            return (
                AccountInfo(
                    balance=broker_info.balance,
                    available_cash=broker_info.available_cash,
                    positions={sym: pos.qty for sym, pos in broker_info.positions.items()},
                    total_exposure=broker_info.total_exposure,
                ),
                broker_info,
                True,
            )
        except BrokerAPIError:
            logger.critical(
                "BROKER API FAILURE: Account info unavailable but auto_trade configured. "
                "This would cause incorrect position sizing. Trade execution disabled for this symbol."
            )
            return (
                AccountInfo(
                    balance=100000.0,  # Mock data - DO NOT USE FOR REAL TRADES
                    available_cash=100000.0,
                    positions={},
                    total_exposure=0.0,
                ),
                None,
                False,
            )  # Signal broker failure

    return await asyncio.to_thread(_sync_get_account)


async def fetch_account_info(broker: AlpacaBroker | None) -> AccountInfoOutput:
    """Fetch account info for portfolio-aware decisions.

    Args:
        broker: Optional Alpaca broker

    Returns:
        AccountInfoOutput with account info and broker positions
    """
    logger.info("Fetching account info")
    account_info, broker_info, account_info_valid = await _get_account_info_internal(broker)

    warnings = []
    broker_api_failed = False

    # Track broker availability for risk assessment
    if not account_info_valid:
        warning = (
            "Broker API unavailable - using mock account data. "
            "Trade execution will be blocked to prevent incorrect position sizing."
        )
        warnings.append(warning)
        broker_api_failed = True

    # Set VaR fields from broker_info (if available)
    broker_positions = None
    portfolio_value = None
    if broker_info:
        broker_positions = broker_info.positions
        portfolio_value = broker_info.portfolio_value

    return AccountInfoOutput(
        account_info=account_info,
        broker_positions=broker_positions,
        portfolio_value=portfolio_value,
        broker_api_failed=broker_api_failed,
        warnings=warnings,
    )
