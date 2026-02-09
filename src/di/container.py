"""DI container for AI Casino application."""

from pathlib import Path

from dependency_injector import containers, providers

from src.di.config import load_daemon_config


class AppContainer(containers.DeclarativeContainer):
    """Application DI container - foundation only.

    Currently provides config - services added in future PRs (#309-#330).
    """

    # Config path storage
    config = providers.Configuration()

    # DaemonConfig singleton - loaded via utility
    daemon_config = providers.Singleton(
        load_daemon_config,
        config_path=config.config_path,
    )


def create_container(config_path: str | Path | None = None) -> AppContainer:
    """Create dependency injection container.

    Args:
        config_path: Optional path to daemon.yaml (supports ~ expansion)

    Returns:
        Configured Container instance
    """
    container = AppContainer()

    if config_path:
        # Expand ~ and resolve to absolute path
        normalized_path = Path(config_path).expanduser().resolve()
        container.config.from_dict({"config_path": normalized_path})

    return container
