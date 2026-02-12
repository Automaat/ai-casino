"""Logging configuration."""

from typing import Literal

from pydantic import BaseModel


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
