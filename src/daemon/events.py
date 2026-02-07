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

    event_id: str = Field(description="Article URL hash")
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
                f"Content: {self.viral_post.content[:500]}\n"
                f"URL: {self.viral_post.url}"
            )
        return (
            f"SOCIAL VOLUME SPIKE:\n"
            f"Symbol: {self.symbol}\n"
            f"Mentions: {self.mention_count} (+{self.mention_delta_pct:.1f}%)\n"
            f"Source: {self.source}"
        )

    def __repr__(self) -> str:
        """String representation."""
        if self.viral_post:
            return f"SocialEvent(viral_post={self.viral_post.title[:30]}...)"
        return f"SocialEvent(symbol={self.symbol}, delta={self.mention_delta_pct:.1f}%)"


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


class AnomalyEvent(BaseModel):
    """Market data anomaly event (volume spike, price move, gap)."""

    event_id: str
    event_type: Literal["anomaly"] = "anomaly"
    timestamp: datetime
    source: str = "market_data"
    symbol: str
    anomaly_type: Literal["volume_spike", "price_move", "gap"]
    volume_ratio: float | None = Field(default=None, description="Current volume / avg volume")
    price_change_pct: float | None = Field(default=None, description="Intraday price change %")

    def to_prompt_text(self) -> str:
        """Format anomaly event for triage."""
        details = []
        if self.volume_ratio:
            details.append(f"Volume: {self.volume_ratio:.1f}x average")
        if self.price_change_pct:
            details.append(f"Price change: {self.price_change_pct:+.1f}%")

        return (
            f"MARKET ANOMALY:\n"
            f"Symbol: {self.symbol}\n"
            f"Type: {self.anomaly_type}\n"
            f"Details: {', '.join(details)}"
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"AnomalyEvent(symbol={self.symbol}, type={self.anomaly_type})"


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

    event: NewsEvent | SocialEvent | FilingEvent | AnomalyEvent = Field(discriminator="event_type")
    triage: TriageResult
    analyses: dict[str, TradingWorkflowResult] = Field(description="Symbol -> analysis result")
    signal_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        """String representation."""
        return f"EventSignal(event={self.event}, symbols={list(self.analyses.keys())})"
