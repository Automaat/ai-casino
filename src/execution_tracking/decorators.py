"""Decorators for automatic execution tracking."""

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast, overload

from src.execution_tracking.context import atrack_execution, track_execution
from src.execution_tracking.models import ExecutionNodeType

T = TypeVar("T")


@overload
def track_agent(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]: ...


@overload
def track_agent(func: Callable[..., T]) -> Callable[..., T]: ...


def track_agent(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to track agent method execution.

    Automatically wraps agent methods (analyze/decide/research/triage) with execution tracking.
    Works with both sync and async methods.

    Args:
        func: Method to wrap

    Returns:
        Wrapped method with execution tracking

    Example:
        ```python
        class TechnicalAnalyst:
            @track_agent
            async def analyze(self, symbol: str, data: pd.DataFrame) -> TechnicalAnalysis:
                ...
        ```
    """
    # Get agent class name from function qualname (e.g., "TechnicalAnalyst.analyze" -> "TechnicalAnalyst")
    agent_name = func.__qualname__.rsplit(".", 1)[0] if "." in func.__qualname__ else "UnknownAgent"

    # Defer signature inspection to call time to avoid circular imports
    sig = None

    # Determine if async
    is_async = inspect.iscoroutinefunction(func)

    if is_async:
        async_func = cast("Callable[..., Awaitable[Any]]", func)

        @functools.wraps(async_func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Async wrapper with execution tracking."""
            # Get signature on first call (defer to avoid circular imports)
            nonlocal sig
            if sig is None:
                sig = inspect.signature(async_func)

            # Extract symbol from args/kwargs if present
            symbol = _extract_symbol(sig, args, kwargs)

            metadata = {"agent": agent_name, "method": async_func.__name__}
            if symbol:
                metadata["symbol"] = symbol

            async with atrack_execution(ExecutionNodeType.AGENT, agent_name, metadata=metadata):
                result = await async_func(*args, **kwargs)
                return result

        return async_wrapper


    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Sync wrapper with execution tracking."""
        # Get signature on first call (defer to avoid circular imports)
        nonlocal sig
        if sig is None:
            sig = inspect.signature(func)

        # Extract symbol from args/kwargs if present
        symbol = _extract_symbol(sig, args, kwargs)

        metadata = {"agent": agent_name, "method": func.__name__}
        if symbol:
            metadata["symbol"] = symbol

        with track_execution(ExecutionNodeType.AGENT, agent_name, metadata=metadata):
            return func(*args, **kwargs)

    return sync_wrapper


def _extract_symbol(sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """Extract symbol from function arguments using signature.

    Args:
        sig: Function signature
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Symbol string if found, None otherwise
    """
    # Check kwargs first
    if "symbol" in kwargs:
        return str(kwargs["symbol"])

    # Map positional args to parameter names using signature
    try:
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        if "symbol" in bound.arguments:
            return str(bound.arguments["symbol"])
    except TypeError:
        # Signature binding failed, skip extraction
        pass

    return None
