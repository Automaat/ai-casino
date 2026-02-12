"""Infrastructure configuration for API, LLM, data sources, and database."""

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator


class PrefetchConfig(BaseModel):
    """Configuration for after-hours data prefetching."""

    enabled: bool = False
    prefetch_time: str = "16:30"
    enable_pre_market_refresh: bool = False
    pre_market_refresh_time: str = "04:00"
    cache_dir: str = "data/cache/prefetch"
    warm_finbert: bool = True
    check_connectivity: bool = True


class ApiConfig(BaseModel):
    """Configuration for embedded API server."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(
        default=8484,
        ge=1,
        le=65535,
        description="TCP port for embedded API server (1-65535)",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8050"],
        description="CORS allowed origins for dashboard access",
    )

    @field_validator("cors_origins")
    @classmethod
    def warn_permissive_cors(cls, v: list[str]) -> list[str]:
        """Warn if CORS origins include wildcards or non-localhost."""
        for origin in v:
            if origin == "*":
                logger.warning(
                    "CORS origin '*' allows all domains - security risk. "
                    "Only use for development in trusted environments."
                )
            elif not any(localhost in origin for localhost in ("localhost", "127.0.0.1", "::1")):
                logger.warning(
                    f"CORS origin '{origin}' is not localhost - allows external dashboard access. "
                    "Only use for development in trusted environments."
                )
        return v

    @field_validator("host")
    @classmethod
    def warn_non_localhost(cls, v: str) -> str:
        """Warn if API binds to non-localhost (security risk)."""
        if v not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                f"API host '{v}' is not localhost - daemon exposed to network without auth. "
                "Only use for development in trusted environments."
            )
        return v


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str | None = None
    model: str | None = None


class DataSourcesConfig(BaseModel):
    """Data sources configuration."""

    market_data: Literal["yfinance", "alpha_vantage"] = "yfinance"


class ApiKeysConfig(BaseModel):
    """API keys configuration.

    All fields are optional and fall back to environment variables.
    Config values take priority when both config and env vars are set.
    """

    alpha_vantage_api_key: str | None = None
    marketaux_api_key: str | None = None
    newsdata_api_key: str | None = None
    finnhub_api_key: str | None = None
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    alpaca_paper_api_key: str | None = None
    alpaca_paper_secret_key: str | None = None
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_api_base: str | None = None


class DatabaseConfig(BaseModel):
    """Database configuration for PostgreSQL persistence."""

    database_url: str | None = None
    pool_size: int = Field(default=5, ge=1, le=20, description="Database connection pool size (1-20)")
    max_overflow: int = Field(default=10, ge=0, le=50, description="Max connections beyond pool_size (0-50)")
    pool_pre_ping: bool = Field(default=True, description="Verify connections before use")
    enable_persistence: bool = Field(default=True, description="Enable database persistence")
