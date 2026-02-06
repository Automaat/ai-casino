"""Retry decorator for LLM provider API calls."""

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

import httpx
from loguru import logger

P = ParamSpec("P")
T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (
        httpx.ConnectError,
        httpx.TimeoutException,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.HTTPStatusError,
    ),
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for retrying async functions on transient errors.

    Args:
        max_attempts: Maximum retry attempts
        delay: Base delay between retries (exponential backoff)
        exceptions: Exception types to retry on
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (attempt + 1)
                        logger.warning(f"Retry {attempt + 1}/{max_attempts} after {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
            if last_error is not None:
                raise last_error
            msg = "Retry decorator invoked with max_attempts <= 0 or no exceptions captured"
            raise RuntimeError(msg)

        return wrapper

    return decorator
