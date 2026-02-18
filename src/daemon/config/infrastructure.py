"""Infrastructure configuration for API, LLM, data sources, and database."""

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.circuit_breaker.models import CircuitBreakerConfig


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
        default_factory=lambda: ["http://localhost:5173", "http://localhost:8050"],
        description="CORS allowed origins for dashboard access",
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration for external API calls",
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

    provider: str = "ollama"
    model: str = "qwen3:14b"
    max_concurrent: int = Field(default=5, ge=1, le=20)
    enable_prompt_caching: bool = Field(
        default=True,
        description="Enable prompt caching (Anthropic: explicit, OpenAI/OpenRouter: metrics only)",
    )
    ollama_base_url: str = "http://localhost:11434"


class FinnhubSourcesConfig(BaseModel):
    """Finnhub premium feature toggles."""

    enable_social_sentiment: bool = False
    enable_news_sentiment: bool = False


class DataSourcesConfig(BaseModel):
    """Data sources configuration."""

    market_data: Literal["yfinance", "alpha_vantage"] = "yfinance"
    finnhub_premium: FinnhubSourcesConfig = Field(default_factory=FinnhubSourcesConfig)


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
    fred_api_key: str | None = None  # FRED API (free: fred.stlouisfed.org)


class FinBERTConfig(BaseModel):
    """FinBERT sentiment service configuration."""

    mode: Literal["local", "remote"] = Field(
        default="local",
        description="FinBERT mode: 'local' (in-process) or 'remote' (microservice)",
    )
    service_url: str = Field(
        default="http://localhost:8485",
        description="FinBERT service URL (only used in remote mode)",
    )
    timeout: float = Field(
        default=60.0,
        ge=10.0,
        le=120.0,
        description="HTTP request timeout in seconds (10-120)",
    )
    workers: int | None = Field(
        default=None,
        ge=1,
        le=32,
        description="ProcessPoolExecutor workers for local mode (1-32, default: os.cpu_count())",
    )


class DatabaseConfig(BaseModel):
    """Database configuration for PostgreSQL persistence."""

    database_url: str | None = None
    pool_size: int = Field(default=5, ge=1, le=20, description="Persistent connections (1-20)")
    max_overflow: int = Field(default=10, ge=0, le=50, description="Burst capacity (0-50)")
    pool_pre_ping: bool = Field(default=True, description="Verify connection health")
    pool_timeout: float = Field(
        default=30.0,
        ge=10.0,
        le=120.0,
        description="Seconds to wait for connection from pool (10-120)",
    )
    pool_recycle: int = Field(
        default=3600,
        ge=300,
        le=7200,
        description="Recycle connections after N seconds to prevent stale (300-7200)",
    )
    enable_persistence: bool = Field(default=True, description="Enable database persistence")
