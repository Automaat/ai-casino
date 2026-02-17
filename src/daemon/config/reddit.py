"""Reddit scraper configuration."""

from dataclasses import dataclass, field


@dataclass
class RedditScraperConfig:
    """Configuration for Reddit web scraping."""

    enabled: bool = True
    use_playwright: bool = True

    # Subreddits
    high_priority_subreddits: list[str] = field(
        default_factory=lambda: ["wallstreetbets", "stocks", "Daytrading"]
    )

    # Scraping limits
    posts_per_subreddit: int = 50
    comments_per_post: int = 10

    # Anti-detection delays (seconds)
    delay_page_load_min: float = 2.0
    delay_page_load_max: float = 8.0
    delay_action_min: float = 0.5
    delay_action_max: float = 2.0

    # Viewport randomization
    viewport_width_min: int = 1280
    viewport_width_max: int = 1920
    viewport_height_min: int = 720
    viewport_height_max: int = 1080

    # Browser settings
    use_stealth_mode: bool = True
    headless: bool = True

    # Scheduling
    interval_minutes: int = 60
    jitter_minutes: int = 15

    # Discovery thresholds
    min_mentions_for_trending: int = 25
    mention_velocity_threshold: float = 2.0  # 200% increase

    # LLM extraction
    use_llm_extraction: bool = True
    extraction_model: str = "haiku"
    extraction_temperature: float = 0.3
    extraction_timeout_s: float = 10.0
    extraction_min_confidence: float = 0.7
    extraction_max_tokens: int = 2000
