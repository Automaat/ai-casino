"""Event models for real-time market signal watcher.

This module defines unified event schemas for all event types (news, social, filings, anomalies)
and triage results from the LLM-powered EventTriageAgent.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from src.data.news import NewsArticle
from src.data.reddit import RedditPost
from src.data.truth_social import TruthPost
from src.workflows.types import TradingWorkflowResult


class Urgency(StrEnum):
    """Event urgency level after triage."""

    IMMEDIATE = "IMMEDIATE"  # Trigger analysis now
    WATCHLIST = "WATCHLIST"  # Monitor but don't analyze
    IGNORE = "IGNORE"  # Skip event


class Sentiment(StrEnum):
    """Event sentiment direction."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class BaseEvent(Protocol):
    """Protocol for all event types."""

    event_id: str
    event_type: str
    timestamp: datetime
    source: str

    def to_prompt_text(self) -> str:
        """Format event for LLM triage prompt."""
        ...


class NewsEvent(BaseModel):
    """News article event."""

    event_id: str = Field(description="Article URL")
    event_type: Literal["news"] = "news"
    timestamp: datetime
    source: str = Field(description="marketaux or duckduckgo")
    article: NewsArticle

    def to_prompt_text(self) -> str:
        """Format news event for triage."""
        return (
            f"NEWS ARTICLE:\n"
            f"Source: {self.article.source}\n"
            f"Title: {self.article.title}\n"
            f"Published: {self.article.published_at}\n"
            f"Description: {self.article.description}\n"
            f"URL: {self.article.url}"
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"NewsEvent(source={self.source}, title={self.article.title[:50]}...)"


class SocialEvent(BaseModel):
    """Social media event (Reddit volume spike or viral post)."""

    event_id: str
    event_type: Literal["social"] = "social"
    timestamp: datetime
    source: str = Field(description="reddit or finnhub")
    symbol: str | None = Field(default=None, description="Stock ticker if volume spike")
    mention_count: int | None = Field(default=None, description="Current mentions")
    mention_delta_pct: float | None = Field(default=None, description="Percentage increase in mentions")
    viral_post: RedditPost | None = Field(default=None, description="Viral post if detected")

    def to_prompt_text(self) -> str:
        """Format social event for triage."""
        if self.viral_post:
            return (
                f"VIRAL SOCIAL POST:\n"
                f"Subreddit: {self.viral_post.subreddit}\n"
                f"Title: {self.viral_post.title}\n"
                f"Score: {self.viral_post.score} (upvote ratio: {self.viral_post.upvote_ratio:.1%})\n"
                f"Content: {self.viral_post.body[:500]}\n"
                f"URL: {self.viral_post.url}"
            )
        return (
            f"SOCIAL VOLUME SPIKE:\n"
            f"Symbol: {self.symbol}\n"
            f"Mentions: {self.mention_count} (+{self.mention_delta_pct or 0.0:.1f}%)\n"
            f"Source: {self.source}"
        )

    def __repr__(self) -> str:
        """String representation."""
        if self.viral_post:
            return f"SocialEvent(viral_post={self.viral_post.title[:30]}...)"
        return f"SocialEvent(symbol={self.symbol}, delta={self.mention_delta_pct or 0.0:.1f}%)"


class FilingEntry(BaseModel):
    """SEC filing metadata."""

    accession_number: str
    filing_type: str = Field(description="8-K, 4, 13D, etc")
    filing_date: datetime
    company_name: str
    cik: str
    url: str

    def __repr__(self) -> str:
        """String representation."""
        return f"FilingEntry(type={self.filing_type}, company={self.company_name})"


class FilingEvent(BaseModel):
    """SEC EDGAR filing event."""

    event_id: str = Field(description="Accession number")
    event_type: Literal["filing"] = "filing"
    timestamp: datetime
    source: str = "sec_edgar"
    filing: FilingEntry
    symbol: str

    def to_prompt_text(self) -> str:
        """Format filing event for triage."""
        return (
            f"SEC FILING:\n"
            f"Company: {self.filing.company_name} ({self.symbol})\n"
            f"Filing Type: {self.filing.filing_type}\n"
            f"Filed: {self.filing.filing_date}\n"
            f"URL: {self.filing.url}"
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"FilingEvent(symbol={self.symbol}, type={self.filing.filing_type})"


class VolumeSpike(BaseModel):
    """Volume spike detection data."""

    current_volume: float
    avg_volume_20d: float
    spike_multiplier: float

    def __repr__(self) -> str:
        """String representation."""
        return f"VolumeSpike({self.spike_multiplier:.1f}x)"


class PriceMove(BaseModel):
    """Intraday price move data."""

    open_price: float
    current_price: float
    change_pct: float
    high: float
    low: float

    def __repr__(self) -> str:
        """String representation."""
        return f"PriceMove({self.change_pct:+.1f}%)"


class Gap(BaseModel):
    """Gap detection data."""

    previous_close: float
    open_price: float
    gap_pct: float
    gap_direction: Literal["up", "down"]

    def __repr__(self) -> str:
        """String representation."""
        return f"Gap({self.gap_direction} {abs(self.gap_pct):.1f}%)"


class TrumpEvent(BaseModel):
    """Trump Truth Social post event."""

    event_id: str = Field(description="Post ID")
    event_type: Literal["trump"] = "trump"
    timestamp: datetime
    source: str = "truth_social"
    post: TruthPost

    def to_prompt_text(self) -> str:
        """Format Trump post for triage."""
        return (
            f"TRUMP TRUTH SOCIAL POST:\n"
            f"Posted: {self.post.created_at}\n"
            f"Content: {self.post.content}\n"
            f"Engagement: {self.post.likes} likes, {self.post.reposts} reposts\n"
            f"URL: {self.post.url}"
        )

    def __repr__(self) -> str:
        """String representation."""
        content_preview = self.post.content[:50].replace("\n", " ")
        return f"TrumpEvent(content={content_preview}...)"


class AnomalyEvent(BaseModel):
    """Market data anomaly event (volume spike, price move, gap)."""

    event_id: str
    event_type: Literal["anomaly"] = "anomaly"
    timestamp: datetime
    source: str = "market_data"
    symbol: str
    anomaly_types: list[Literal["volume_spike", "price_move", "gap"]]

    volume_spike_data: VolumeSpike | None = None
    price_move_data: PriceMove | None = None
    gap_data: Gap | None = None

    def to_prompt_text(self) -> str:
        """Format anomaly event for triage."""
        details = []

        if self.volume_spike_data:
            details.append(
                f"Volume Spike: {self.volume_spike_data.current_volume:,.0f} "
                f"({self.volume_spike_data.spike_multiplier:.1f}x 20-day avg of "
                f"{self.volume_spike_data.avg_volume_20d:,.0f})"
            )

        if self.price_move_data:
            details.append(
                f"Price Move: ${self.price_move_data.open_price:.2f} → "
                f"${self.price_move_data.current_price:.2f} "
                f"({self.price_move_data.change_pct:+.1f}%) "
                f"[H: ${self.price_move_data.high:.2f}, L: ${self.price_move_data.low:.2f}]"
            )

        if self.gap_data:
            details.append(
                f"Gap: ${self.gap_data.previous_close:.2f} → ${self.gap_data.open_price:.2f} "
                f"({self.gap_data.gap_direction} {abs(self.gap_data.gap_pct):.1f}%)"
            )

        details_text = "\n".join(f"  - {d}" for d in details)
        return (
            f"MARKET ANOMALY:\n"
            f"Symbol: {self.symbol}\n"
            f"Anomaly Types: {', '.join(self.anomaly_types)}\n"
            f"Details:\n{details_text}"
        )

    def __repr__(self) -> str:
        """String representation."""
        types_str = "+".join(self.anomaly_types)
        return f"AnomalyEvent(symbol={self.symbol}, types={types_str})"


class NewsTrendingEvent(BaseModel):
    """News trending event (symbol appeared frequently in recent news)."""

    event_id: str
    event_type: Literal["news_trending"] = "news_trending"
    timestamp: datetime
    source: str = "news_trending_watcher"
    symbol: str
    mention_count: int
    articles: list[str]
    baseline_count: float
    spike_ratio: float
    trending_window: int

    def to_prompt_text(self) -> str:
        """Format for LLM triage."""
        titles = "\n".join(f"  - {t}" for t in self.articles[:5])
        return (
            f"NEWS TRENDING:\n"
            f"Symbol: {self.symbol}\n"
            f"Mentioned in {self.mention_count} articles (last {self.trending_window}min)\n"
            f"Baseline: {self.baseline_count:.1f} mentions/hour\n"
            f"Spike ratio: {self.spike_ratio:.1f}x\n"
            f"Recent headlines:\n{titles}"
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NewsTrendingEvent(symbol={self.symbol}, "
            f"mentions={self.mention_count}, spike={self.spike_ratio:.1f}x)"
        )


class EconomicImpact(StrEnum):
    """Economic event impact level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EconomicRiskLevel(StrEnum):
    """Risk level from economic calendar assessment."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EconomicRecommendation(StrEnum):
    """Trading recommendation based on economic risk."""

    TRADE_NORMALLY = "trade_normally"
    REDUCE_SIZE = "reduce_size"
    AVOID_NEW_POSITIONS = "avoid_new_positions"


class EconomicEvent(BaseModel):
    """Single economic calendar entry."""

    event_id: str  # f"{country}_{event}_{time}"
    country: str
    event: str
    impact: EconomicImpact
    scheduled_at: datetime
    actual: str | None = None
    estimate: str | None = None
    prev: str | None = None

    def __repr__(self) -> str:
        """String representation."""
        return f"EconomicEvent(event={self.event}, impact={self.impact}, at={self.scheduled_at.date()})"


class EconomicEventSignal(BaseModel):
    """Risk assessment from economic calendar watcher."""

    upcoming_events: list[EconomicEvent]
    risk_level: EconomicRiskLevel
    recommendation: EconomicRecommendation
    reason: str
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    avoid_until: datetime | None = None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EconomicEventSignal(risk={self.risk_level}, "
            f"recommendation={self.recommendation}, events={len(self.upcoming_events)})"
        )


class OptionsFlowDirection(StrEnum):
    """Net premium direction from options flow."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class BlockTrade(BaseModel):
    """High-premium options contract detection."""

    strike: float
    expiry: str
    premium: float
    volume: int
    option_type: str = Field(description="call or put")
    is_itm: bool

    def __repr__(self) -> str:
        """String representation."""
        return f"BlockTrade({self.option_type} {self.strike} ${self.premium:,.0f})"


class OptionsFlowSignal(BaseModel):
    """Options flow assessment for a single symbol."""

    symbol: str
    put_call_ratio: float
    volume_vs_avg: float
    has_unusual_activity: bool
    block_trades: list[BlockTrade] = Field(default_factory=list)
    net_premium_direction: OptionsFlowDirection
    significance_score: float = Field(ge=0.0, le=1.0)
    reason: str
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptionsFlowSignal({self.symbol} "
            f"P/C={self.put_call_ratio:.2f} "
            f"dir={self.net_premium_direction} "
            f"score={self.significance_score:.2f})"
        )


class TriageResult(BaseModel):
    """LLM triage result for an event."""

    event_id: str
    event_type: str
    relevance: float = Field(ge=0.0, le=1.0, description="Relevance score 0-1")
    symbols: list[str] = Field(description="Extracted stock tickers")
    urgency: Urgency
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0, description="Triage confidence 0-1")
    reasoning: str = Field(description="Explanation for triage decision")
    triaged_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TriageResult(symbols={self.symbols}, urgency={self.urgency.value}, "
            f"relevance={self.relevance:.2f})"
        )


class EventSignal(BaseModel):
    """Signal emitted after triage + analysis."""

    event: NewsEvent | SocialEvent | TrumpEvent | FilingEvent | AnomalyEvent | NewsTrendingEvent = Field(
        discriminator="event_type"
    )
    triage: TriageResult
    analyses: dict[str, TradingWorkflowResult] = Field(description="Symbol -> analysis result")
    signal_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        """String representation."""
        return f"EventSignal(event={self.event}, symbols={list(self.analyses.keys())})"
