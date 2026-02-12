"""UI and dashboard configuration."""

from pydantic import BaseModel, Field


class UIConfig(BaseModel):
    """UI and dashboard configuration."""

    theme: str | None = None
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = Field(default=8050, ge=1, le=65535)
    daemon_api_url: str = "http://localhost:8484"
    dashboard_refresh_interval: int = Field(default=5000, ge=1000, le=60000)
