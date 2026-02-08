"""Dashboard configuration."""

import os

from pydantic import BaseModel, Field


class DashboardConfig(BaseModel):
    """Dashboard configuration with validation."""

    api_url: str = "http://localhost:8001"
    refresh_interval: int = Field(default=5000, ge=1000, le=60000)
    port: int = Field(default=8050, ge=1, le=65535)
    host: str = Field(default_factory=lambda: os.getenv("DASHBOARD_HOST", "127.0.0.1"))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DashboardConfig(api_url={self.api_url}, port={self.port}, "
            f"refresh_interval={self.refresh_interval}ms)"
        )
