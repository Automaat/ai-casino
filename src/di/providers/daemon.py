"""Daemon component providers for dependency injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.daemon.config import DaemonConfig
    from src.daemon.context_builder import DaemonContextBuilder
    from src.daemon.factory import DaemonComponents, DaemonFactory
    from src.daemon.task_service import DaemonTaskService
    from src.di.container import AppContainer


def create_daemon_factory(
    daemon_config: DaemonConfig,
    container: AppContainer,
) -> DaemonFactory:
    """Create DaemonFactory with resolved config and container.

    Args:
        daemon_config: Daemon configuration
        container: DI container for service resolution

    Returns:
        DaemonFactory instance
    """
    from src.daemon.factory import DaemonFactory

    return DaemonFactory(daemon_config, container=container)


def create_context_builder(
    components: DaemonComponents,
    container: AppContainer,
) -> DaemonContextBuilder:
    """Create DaemonContextBuilder with components and container.

    Args:
        components: Daemon components
        container: DI container for fetcher access

    Returns:
        DaemonContextBuilder instance
    """
    from src.daemon.context_builder import DaemonContextBuilder

    return DaemonContextBuilder(components, container)


def create_task_service(
    components: DaemonComponents,
    container: AppContainer,
) -> DaemonTaskService:
    """Create DaemonTaskService with components and container.

    Args:
        components: Daemon components
        container: DI container for service access

    Returns:
        DaemonTaskService instance
    """
    from src.daemon.task_service import DaemonTaskService

    return DaemonTaskService(components, container)
