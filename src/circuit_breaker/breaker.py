"""Circuit breaker implementation."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import ParamSpec, TypeVar

import httpx
from loguru import logger

from src.circuit_breaker.models import (
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerState,
    CircuitBreakerStatus,
)

# HTTP status codes for error categorization
HTTP_PAYMENT_REQUIRED = 402
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_GATEWAY_TIMEOUT = 504

P = ParamSpec("P")
T = TypeVar("T")


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, service: str, config: CircuitBreakerConfig) -> None:
        """Initialize circuit breaker.

        Args:
            service: Service name (e.g., "marketaux")
            config: Circuit breaker configuration
        """
        self._service = service
        self._config = config
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: datetime | None = None
        self._error_code: int | None = None  # Tracks error that opened circuit
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        logger.info(f"Initialized circuit breaker for {service}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CircuitBreaker(service={self._service}, state={self._state})"

    @property
    def config(self) -> CircuitBreakerConfig:
        """Get circuit breaker configuration."""
        return self._config

    async def call(self, func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            await self._check_state()

        was_half_open = self._state == CircuitBreakerState.HALF_OPEN
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            # Decrement half-open calls if exception occurs in HALF_OPEN state
            if was_half_open:
                async with self._lock:
                    self._half_open_calls = max(0, self._half_open_calls - 1)

            if self._is_retriable_error(e):
                await self._on_failure(e)
            raise

    async def _check_state(self) -> None:
        """Check circuit state and raise if open."""
        if self._state == CircuitBreakerState.CLOSED:
            return

        if self._state == CircuitBreakerState.HALF_OPEN:
            if self._half_open_calls >= self._config.half_open_max_calls:
                open_until = self._calculate_open_until()
                raise CircuitBreakerError(self._service, open_until)
            self._half_open_calls += 1
            return

        # OPEN state - check timeout
        if self._opened_at is None:
            return

        open_until = self._calculate_open_until()
        if datetime.now(UTC) >= open_until:
            await self._transition_to_half_open()
        else:
            raise CircuitBreakerError(self._service, open_until)

    def _calculate_open_until(self) -> datetime:
        """Calculate when circuit will transition to half-open based on error type."""
        if self._opened_at is None:
            return datetime.now(UTC)

        # Use error-specific timeout if available
        if self._error_code == HTTP_PAYMENT_REQUIRED:
            timeout = self._config.quota_error_timeout_seconds
        elif self._error_code == HTTP_TOO_MANY_REQUESTS:
            timeout = self._config.rate_limit_timeout_seconds
        elif self._error_code in (HTTP_SERVICE_UNAVAILABLE, HTTP_GATEWAY_TIMEOUT):
            timeout = self._config.server_error_timeout_seconds
        else:
            timeout = self._config.timeout_seconds

        return self._opened_at + timedelta(seconds=timeout)

    async def _on_success(self) -> None:
        """Handle successful request."""
        async with self._lock:
            self._failure_count = 0

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                self._half_open_calls = max(0, self._half_open_calls - 1)

                if self._success_count >= self._config.success_threshold:
                    await self._transition_to_closed()
                    logger.info(f"Circuit closed for {self._service}")

    async def _on_failure(self, exc: Exception) -> None:
        """Handle failed request.

        Args:
            exc: Exception that caused the failure
        """
        async with self._lock:
            self._failure_count += 1

            # Extract error code for timeout calculation
            error_code = self._extract_error_code(exc)

            if self._state == CircuitBreakerState.HALF_OPEN:
                await self._transition_to_open(error_code)
                logger.warning(f"Circuit reopened for {self._service} after failed test (error {error_code})")
            elif self._failure_count >= self._config.failure_threshold:
                await self._transition_to_open(error_code)
                timeout = self._calculate_timeout_for_error(error_code)
                logger.warning(
                    f"Circuit opened for {self._service} after {self._failure_count} failures "
                    f"(error {error_code}, timeout {timeout}s)"
                )

    async def _transition_to_open(self, error_code: int | None = None) -> None:
        """Transition to OPEN state.

        Args:
            error_code: HTTP status code that triggered the transition
        """
        self._state = CircuitBreakerState.OPEN
        self._opened_at = datetime.now(UTC)
        self._error_code = error_code
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self._state = CircuitBreakerState.HALF_OPEN
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        logger.info(f"Circuit transitioned to HALF_OPEN for {self._service}")

    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self._state = CircuitBreakerState.CLOSED
        self._opened_at = None
        self._error_code = None
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

    def _extract_error_code(self, exc: Exception) -> int | None:
        """Extract HTTP status code from exception.

        Args:
            exc: Exception to extract code from

        Returns:
            HTTP status code or None
        """
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code
        return None

    def _calculate_timeout_for_error(self, error_code: int | None) -> int:
        """Calculate timeout based on error code category.

        Args:
            error_code: HTTP status code

        Returns:
            Timeout in seconds
        """
        if error_code == HTTP_PAYMENT_REQUIRED:
            return self._config.quota_error_timeout_seconds
        if error_code == HTTP_TOO_MANY_REQUESTS:
            return self._config.rate_limit_timeout_seconds
        if error_code in (HTTP_SERVICE_UNAVAILABLE, HTTP_GATEWAY_TIMEOUT):
            return self._config.server_error_timeout_seconds
        return self._config.timeout_seconds

    def _is_retriable_error(self, exc: Exception) -> bool:
        """Check if error should affect circuit state."""
        if isinstance(exc, httpx.HTTPStatusError):
            # Payment Required, Too Many Requests, server errors
            return exc.response.status_code in (
                HTTP_PAYMENT_REQUIRED,
                HTTP_TOO_MANY_REQUESTS,
                HTTP_SERVICE_UNAVAILABLE,
                HTTP_GATEWAY_TIMEOUT,
            )

        return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout))

    def get_status(self) -> CircuitBreakerStatus:
        """Get current circuit breaker status."""
        status = CircuitBreakerStatus(
            service=self._service,
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            opened_at=self._opened_at,
            half_open_calls=self._half_open_calls,
            error_code=self._error_code,
        )

        # Calculate open_until for status
        if self._state == CircuitBreakerState.OPEN and self._opened_at is not None:
            status.open_until = self._calculate_open_until()

        return status
