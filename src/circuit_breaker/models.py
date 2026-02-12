"""Circuit breaker models and configuration."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class CircuitBreakerState(StrEnum):
    """Circuit breaker state."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Blocking requests
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    failure_threshold: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Consecutive failures before opening",
    )
    success_threshold: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Successes in half-open to close",
    )
    timeout_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Default timeout before testing recovery",
    )
    quota_error_timeout_seconds: int = Field(
        default=3600,
        ge=300,
        le=172800,
        description="Timeout for quota errors (402) - default 1h",
    )
    rate_limit_timeout_seconds: int = Field(
        default=300,
        ge=60,
        le=86400,
        description="Timeout for rate limits (429) - default 5m",
    )
    server_error_timeout_seconds: int = Field(
        default=60,
        ge=30,
        le=3600,
        description="Timeout for server errors (5xx) - default 1m",
    )
    half_open_max_calls: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Max concurrent test requests",
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CircuitBreakerConfig(failure_threshold={self.failure_threshold}, "
            f"timeout_seconds={self.timeout_seconds})"
        )


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, service: str, open_until: datetime) -> None:
        """Initialize circuit breaker error.

        Args:
            service: Service name
            open_until: When circuit will transition to half-open
        """
        self.service = service
        self.open_until = open_until
        super().__init__(f"Circuit breaker open for {service} until {open_until}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CircuitBreakerError(service={self.service}, open_until={self.open_until})"


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status snapshot."""

    service: str
    state: CircuitBreakerState
    failure_count: int
    success_count: int
    opened_at: datetime | None = None
    half_open_calls: int = 0
    open_until: datetime | None = None
    error_code: int | None = None  # HTTP status code that triggered OPEN state

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CircuitBreakerStatus(service={self.service}, state={self.state})"
