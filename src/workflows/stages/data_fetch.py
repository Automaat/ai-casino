"""Data fetch stage implementation."""

from __future__ import annotations

import asyncio
import zoneinfo
from collections.abc import Coroutine
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.data.broker import AlpacaBroker
    from src.data.market import MarketDataFetcher
    from src.data.news import NewsFetcher
    from src.data.truth_social import TruthSocialFetcher

from src.agents.risk import AccountInfo
from src.data.broker import BrokerAccountInfo, BrokerAPIError
from src.data.market import MarketData
from src.data.truth_social import TrumpPostData
from src.strategies.session import TradingSession
from src.strategies.timeframe import MultiTimeframeData, Timeframe
from src.workflows.models.account import AccountInfoOutput
from src.workflows.models.data_fetch import FetchDataOutput

ET_TIMEZONE = zoneinfo.ZoneInfo("America/New_York")
MARKET_HOURS_START = 4
MARKET_HOURS_END = 20


def _is_market_hours() -> bool:
    """Check if currently within market hours (4am-8pm ET)."""
    now = datetime.now(ET_TIMEZONE)
    return MARKET_HOURS_START <= now.hour < MARKET_HOURS_END


async def fetch_data(
    symbol: str,
    period_days: int,
    trading_session: TradingSession,
    market_fetcher: MarketDataFetcher,
    news_fetcher: NewsFetcher,
    enable_multi_timeframe: bool = False,
    trump_mode: bool = False,
    trump_fetcher: TruthSocialFetcher | None = None,
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

    Returns:
        FetchDataOutput with market and news data
    """
    logger.info("Fetching market and news data")

    # Capture market hours decision once to avoid race condition
    use_multi_timeframe = enable_multi_timeframe and _is_market_hours()

    # Prepare parallel tasks
    if use_multi_timeframe:
        logger.info("Multi-timeframe mode enabled (market hours)")
        market_task = market_fetcher.fetch_multi_timeframe(
            symbol, [Timeframe.DAILY, Timeframe.HOURLY], period_days
        )
    else:
        if enable_multi_timeframe and not use_multi_timeframe:
            logger.info("Multi-timeframe requested but outside market hours, using daily only")
        market_task = asyncio.to_thread(market_fetcher.fetch_daily, symbol, period_days)

    news_task = asyncio.to_thread(news_fetcher.fetch_company_news, symbol, limit=10)
    tasks: list[Coroutine[Any, Any, Any]] = [market_task, news_task]

    if trump_mode and trump_fetcher:
        tasks.append(asyncio.to_thread(trump_fetcher.fetch_recent, hours=24))

    # Execute in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Extract market data
    if isinstance(results[0], Exception):
        logger.error(f"Market data fetch failed: {results[0]}")
        raise results[0]
    market_result = results[0]
    if use_multi_timeframe:
        # fetch_multi_timeframe returns MultiTimeframeData
        assert isinstance(market_result, MultiTimeframeData)  # noqa: S101
        market_data = market_result
    else:
        # fetch_daily returns MarketData with .data attribute
        assert isinstance(market_result, MarketData)  # noqa: S101
        market_data = market_result.data

    # Extract news data
    if isinstance(results[1], Exception):
        logger.warning(f"News fetch failed, continuing with empty news: {results[1]}")
        news_result = []
    else:
        news_result = results[1]
        assert isinstance(news_result, list)  # noqa: S101

    # Extract trump data
    trump_posts = None
    if trump_mode and trump_fetcher:
        trump_result = results[2]
        if isinstance(trump_result, Exception):
            logger.warning(f"Failed to fetch Trump posts: {trump_result}")
        else:
            assert isinstance(trump_result, TrumpPostData)  # noqa: S101
            trump_posts = trump_result.posts
            logger.info(f"Fetched {len(trump_posts)} Trump posts")

    return FetchDataOutput(
        symbol=symbol,
        trading_session=trading_session,
        market_data=market_data,
        news_articles=news_result,
        trump_posts=trump_posts,
        enable_multi_timeframe=enable_multi_timeframe,
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
