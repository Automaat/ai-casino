"""Reddit scraper configuration."""

from pydantic import BaseModel, Field


class RedditScraperConfig(BaseModel):
    """Configuration for Reddit web scraping."""

    enabled: bool = True
    use_playwright: bool = True

    # Subreddits
    high_priority_subreddits: list[str] = Field(
        default_factory=lambda: ["wallstreetbets", "stocks", "Daytrading"]
    )

    # Scraping limits
    posts_per_subreddit: int = Field(default=50, ge=1, le=100)
    comments_per_post: int = Field(default=10, ge=1, le=50)

    # Anti-detection delays (seconds) — old.reddit.com has minimal anti-bot
    delay_page_load_min: float = Field(default=1.0, ge=0.0)
    delay_page_load_max: float = Field(default=3.0, ge=0.0)
    delay_action_min: float = Field(default=0.3, ge=0.0)
    delay_action_max: float = Field(default=1.0, ge=0.0)

    # Viewport randomization
    viewport_width_min: int = Field(default=1280, ge=800, le=3840)
    viewport_width_max: int = Field(default=1920, ge=800, le=3840)
    viewport_height_min: int = Field(default=720, ge=600, le=2160)
    viewport_height_max: int = Field(default=1080, ge=600, le=2160)

    # Browser settings
    use_stealth_mode: bool = True
    headless: bool = True

    # Scheduling
    interval_minutes: int = Field(default=60, ge=1)
    jitter_minutes: int = Field(default=15, ge=0)

    # LLM extraction
    use_llm_extraction: bool = True
    extraction_model: str = "haiku"
    extraction_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    extraction_timeout_s: float = Field(default=10.0, ge=1.0)
    extraction_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    extraction_max_tokens: int = Field(default=2000, ge=100)
