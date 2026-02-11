"""Config loading utilities for DI."""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.daemon.config import DaemonConfig


def load_daemon_config(config_path: Path | None = None) -> DaemonConfig:
    """Load daemon config from YAML or return defaults.

    Loads .env first (for downstream env var consumers), then loads
    DaemonConfig from YAML if path provided. Returns default DaemonConfig()
    if no path given.

    Args:
        config_path: Path to daemon.yaml, or None for defaults

    Returns:
        DaemonConfig from YAML or default instance

    Raises:
        FileNotFoundError: If config_path provided but doesn't exist
    """
    load_dotenv()  # Idempotent - safe to call multiple times

    if config_path is None:
        logger.debug("No config_path - returning default DaemonConfig")
        return DaemonConfig()

    if not config_path.exists():
        msg = f"Config not found: {config_path}"
        raise FileNotFoundError(msg)

    logger.info(f"Loading daemon config from {config_path}")
    return DaemonConfig.from_yaml(config_path)


def resolve_config_or_env(config_value: str | None, env_var: str) -> str | None:
    """Resolve from config or env var. Config priority.

    Matches existing _resolve_config_or_env() pattern in DaemonRunner.
    For future migrations.

    Args:
        config_value: YAML config value
        env_var: Env var name

    Returns:
        config_value if truthy, else env var value
    """
    return config_value or os.getenv(env_var)
