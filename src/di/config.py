"""Config loading utilities for DI."""

from pathlib import Path

from loguru import logger

from src.daemon.config import DaemonConfig


def load_daemon_config(config_path: Path | None = None) -> DaemonConfig:
    """Load daemon config from YAML or return defaults.

    Args:
        config_path: Path to daemon.yaml, or None for defaults

    Returns:
        DaemonConfig from YAML or default instance

    Raises:
        FileNotFoundError: If config_path provided but doesn't exist
    """
    if config_path is None:
        logger.debug("No config_path - returning default DaemonConfig")
        return DaemonConfig()

    if not config_path.exists():
        msg = f"Config not found: {config_path}"
        raise FileNotFoundError(msg)

    logger.info(f"Loading daemon config from {config_path}")
    return DaemonConfig.from_yaml(config_path)
