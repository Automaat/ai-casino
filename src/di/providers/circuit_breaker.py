"""Circuit breaker DI providers."""

from loguru import logger

from src.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from src.daemon.config import DaemonConfig


def create_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Create singleton circuit breaker registry.

    Returns:
        CircuitBreakerRegistry instance
    """
    logger.debug("Creating circuit breaker registry")
    return CircuitBreakerRegistry()


async def create_circuit_breaker(
    service: str,
    daemon_config: DaemonConfig,
    registry: CircuitBreakerRegistry,
) -> CircuitBreaker:
    """Get or create circuit breaker for service.

    Args:
        service: Service name (e.g., "marketaux")
        daemon_config: Daemon configuration
        registry: Circuit breaker registry

    Returns:
        CircuitBreaker instance for service
    """
    config = daemon_config.api.circuit_breaker
    logger.debug(f"Getting circuit breaker for {service} with config: {config}")
    return await registry.get_breaker(service, config)
