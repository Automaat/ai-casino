"""Service configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """FinBERT service settings."""

    device: str = Field(default="cpu", description="Device for inference: cpu or cuda")
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    port: int = Field(default=8485, description="Service port")

    class Config:
        env_prefix = "FINBERT_"
