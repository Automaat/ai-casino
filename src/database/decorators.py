"""Decorators for database operations."""

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from loguru import logger

if TYPE_CHECKING:
    from src.database.repositories.base import BaseRepository

T = TypeVar("T")
P = ParamSpec("P")


def handle_event_loop_error(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Decorator to handle event loop binding errors.

    When a repository method is called from a different event loop than where
    the session was created, catch the RuntimeError and recreate the session
    in the current event loop.

    Args:
        func: Async repository method to wrap

    Returns:
        Wrapped method with error handling
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # First arg is self (BaseRepository instance)
        self = cast("BaseRepository[Any]", args[0]) if args else None

        try:
            return await func(*args, **kwargs)
        except RuntimeError as e:
            if "bound to a different event loop" in str(e) and self is not None:
                logger.warning(f"{func.__name__} encountered event loop error, recreating session")
                # Recreate session in current event loop
                from src.database.connection import get_db_engine

                engine = get_db_engine()
                self._session = engine.session()

                # Retry operation with new session
                return await func(*args, **kwargs)
            raise
        except Exception as e:
            logger.opt(exception=True).error(f"Database operation failed: {e}")
            raise

    return wrapper
