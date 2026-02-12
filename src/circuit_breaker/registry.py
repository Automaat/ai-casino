"""Circuit breaker registry for centralized management."""

import asyncio

from loguru import logger

from src.circuit_breaker.breaker import CircuitBreaker
from src.circuit_breaker.models import CircuitBreakerConfig, CircuitBreakerStatus


class CircuitBreakerRegistry:
    """Centralized circuit breaker management."""

    def __init__(self) -> None:
        """Initialize circuit breaker registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
        logger.info("Initialized circuit breaker registry")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CircuitBreakerRegistry(breakers={len(self._breakers)})"

    async def get_breaker(
        self,
        service: str,
        config: CircuitBreakerConfig,
    ) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        async with self._lock:
            if service not in self._breakers:
                self._breakers[service] = CircuitBreaker(service, config)
                logger.debug(f"Created circuit breaker for {service}")

            return self._breakers[service]

    def get_all_statuses(self) -> dict[str, CircuitBreakerStatus]:
        """Get status of all circuit breakers."""
        return {service: breaker.get_status() for service, breaker in self._breakers.items()}

    def get_status(self, service: str) -> CircuitBreakerStatus | None:
        """Get status of specific circuit breaker."""
        breaker = self._breakers.get(service)
        return breaker.get_status() if breaker else None

    async def reset(self, service: str) -> bool:
        """Reset circuit breaker to closed state."""
        async with self._lock:
            if service not in self._breakers:
                return False

            # Create new breaker with same config (resets state)
            old_breaker = self._breakers[service]
            config = old_breaker.config
            self._breakers[service] = CircuitBreaker(service, config)
            logger.info(f"Reset circuit breaker for {service}")
            return True

    async def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        async with self._lock:
            for service in list(self._breakers.keys()):
                old_breaker = self._breakers[service]
                config = old_breaker.config
                self._breakers[service] = CircuitBreaker(service, config)

            logger.info(f"Reset all {len(self._breakers)} circuit breakers")
