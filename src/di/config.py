"""Config loading utilities for DI."""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.daemon.config import DaemonConfig


def load_daemon_config(config_path: Path | None = None) -> DaemonConfig | None:
    """Load DaemonConfig with env + YAML priority.

    Loads .env first, then YAML. Preserves priority: YAML > env > defaults.

    Args:
        config_path: Optional daemon.yaml path

    Returns:
        DaemonConfig if config_path provided, None otherwise
    """
    load_dotenv()  # Idempotent - safe to call multiple times

    if config_path is None:
        logger.debug("No config_path - skipping YAML")
        return None

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
