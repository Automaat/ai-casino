"""Dashboard configuration."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.daemon.config import DaemonConfig


class DashboardConfig(BaseModel):
    """Dashboard configuration with validation."""

    api_url: str = "http://localhost:8484"
    refresh_interval: int = Field(default=5000, ge=1000, le=60000)
    port: int = Field(default=8050, ge=1, le=65535)
    host: str = "127.0.0.1"

    @classmethod
    def from_daemon_config(cls, daemon_config: DaemonConfig) -> DashboardConfig:
        """Create DashboardConfig from DaemonConfig.

        Args:
            daemon_config: Daemon configuration

        Returns:
            DashboardConfig instance
        """
        return cls(
            api_url=daemon_config.ui.daemon_api_url,
            refresh_interval=daemon_config.ui.dashboard_refresh_interval,
            port=daemon_config.ui.dashboard_port,
            host=daemon_config.ui.dashboard_host,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DashboardConfig(api_url={self.api_url}, port={self.port}, "
            f"refresh_interval={self.refresh_interval}ms)"
        )
