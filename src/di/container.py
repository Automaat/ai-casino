"""DI container for AI Casino application."""

from pathlib import Path

from dependency_injector import containers, providers

from src.daemon.config import DaemonConfig


class AppContainer(containers.DeclarativeContainer):
    """Application DI container - foundation only.

    Currently provides config - services added in future PRs (#309-#330).
    """

    # Config path storage
    config = providers.Configuration()

    # DaemonConfig singleton - parsed from YAML
    daemon_config = providers.Singleton(
        DaemonConfig.from_yaml,
        path=config.config_path,
    )


def create_container(config_path: Path | None = None) -> AppContainer:
    """Create configured container.

    Args:
        config_path: Optional daemon.yaml path. If None, daemon_config
                    provider will fail (use for future service-only containers)

    Returns:
        AppContainer instance

    Example:
        container = create_container(Path("daemon.yaml"))
        config = container.daemon_config()  # Pydantic model
    """
    container = AppContainer()

    if config_path:
        container.config.from_dict({"config_path": config_path})

    return container
