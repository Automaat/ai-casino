"""Shared utilities for API routers."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from fastapi import Request
from loguru import logger
from result import Err

if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents

_broker_cache: ContextVar[dict[str, Any] | None] = ContextVar("_broker_cache", default=None)


def get_components(request: Request) -> DaemonComponents:
    """Get DaemonComponents from app state.

    Args:
        request: FastAPI request

    Returns:
        DaemonComponents instance
    """
    return request.app.state.components


def mask_sensitive_field(value: str | None) -> str:
    """Mask sensitive API key (show first 4 + last 4 chars when long enough).

    Args:
        value: API key or None

    Returns:
        Masked string (e.g., "sk-1234...xy89") for values with length >= 8,
        "***" for shorter non-empty values, or "Not set" when value is falsy.
    """
    if not value:
        return "Not set"
    if len(value) < 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


@asynccontextmanager
async def get_broker_account_info_cached(
    components: DaemonComponents,
) -> AsyncIterator[dict[str, Any] | None]:
    """Request-scoped cached broker account info.

    Args:
        components: DaemonComponents instance

    Yields:
        Broker account info dict or None if broker unavailable
    """
    if not components.broker:
        yield None
        return

    token = _broker_cache.set({})
    cache = _broker_cache.get()
    if cache is None:
        msg = "Cache should be set after _broker_cache.set({})"
        raise RuntimeError(msg)
    cache_key = "account_info"

    try:
        if cache_key not in cache:
            try:
                from src.data.broker import BrokerAccountInfo

                account_result = await asyncio.to_thread(components.broker.get_account_info)
                if isinstance(account_result, Err):
                    raise account_result.err_value
                account_info: BrokerAccountInfo = account_result.ok()
                cache[cache_key] = {
                    "positions": account_info.positions,
                    "portfolio_value": account_info.portfolio_value,
                    "balance": account_info.balance,
                }
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to fetch broker account info: {e}")
                cache[cache_key] = None

        yield cache[cache_key]
    finally:
        _broker_cache.reset(token)
