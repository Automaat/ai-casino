"""Adapter to convert event signals to discovery candidates."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from loguru import logger

from src.daemon.events import Urgency
from src.discovery.models import DiscoveryCandidate, DiscoverySource

if TYPE_CHECKING:
    from src.daemon.events import BaseEvent, TriageResult
    from src.data.market import MarketDataFetcher


class EventDiscoveryAdapter:
    """Converts event signals to discovery candidates."""

    def __init__(self, market_fetcher: MarketDataFetcher) -> None:
        """Initialize adapter.

        Args:
            market_fetcher: Market data fetcher for enrichment
        """
        self.market_fetcher = market_fetcher

    async def convert_event_to_candidate(
        self, event: BaseEvent, triage: TriageResult
    ) -> list[DiscoveryCandidate]:
        """Convert triaged event to discovery candidates.

        Maps event type → DiscoverySource
        Enriches with market data (price, volume, ATR)
        Calculates composite score from triage relevance
        Sets TTL based on urgency (IMMEDIATE=4h, WATCHLIST=24h)

        Args:
            event: Event object
            triage: Triage result with symbols and urgency

        Returns:
            List of discovery candidates
        """
        source = self._map_event_to_source(event.event_type)
        ttl_hours = self._get_ttl_hours(triage.urgency)

        candidates = []
        for symbol in triage.symbols:
            market_data = await self._fetch_market_context(symbol)

            composite_score = self._calculate_event_score(
                relevance=triage.relevance,
                urgency=triage.urgency,
                confidence=triage.confidence,
                market_data=market_data,
            )

            ttl_expires_at = datetime.now(UTC) + timedelta(hours=ttl_hours)

            candidate = DiscoveryCandidate(
                symbol=symbol,
                name=str(market_data.get("name", "Unknown")),
                sector=str(market_data.get("sector", "Unknown")),
                sources=[source],
                composite_score=composite_score,
                source_scores={str(source.value): composite_score},
                discovery_timestamp=datetime.now(UTC),
                ttl_expires_at=ttl_expires_at,
                metadata={
                    "event_type": event.event_type,
                    "event_id": event.event_id,
                    "triage_relevance": triage.relevance,
                    "triage_urgency": triage.urgency.value,
                    "triage_sentiment": triage.sentiment.value,
                    **market_data,
                },
            )
            candidates.append(candidate)

        logger.info(
            f"Converted {event.event_type} event to {len(candidates)} candidates "
            f"(urgency={triage.urgency.value}, ttl={ttl_hours}h)"
        )

        return candidates

    def _map_event_to_source(self, event_type: str) -> DiscoverySource:
        """Map event type to discovery source.

        Args:
            event_type: Event type string

        Returns:
            Corresponding DiscoverySource
        """
        event_source_map = {
            "news": DiscoverySource.NEWS_TRENDING,
            "social": DiscoverySource.REDDIT_TRENDING,
            "trump": DiscoverySource.NEWS_TRENDING,
            "anomaly": DiscoverySource.VOLUME_SPIKE,
            "filing": DiscoverySource.NEWS_TRENDING,
        }

        return event_source_map.get(event_type, DiscoverySource.NEWS_TRENDING)

    def _get_ttl_hours(self, urgency: Urgency) -> int:
        """Get TTL hours based on urgency.

        Args:
            urgency: Event urgency level

        Returns:
            TTL in hours
        """
        ttl_map = {
            Urgency.IMMEDIATE: 4,
            Urgency.WATCHLIST: 24,
            Urgency.IGNORE: 0,
        }

        return ttl_map.get(urgency, 24)

    async def _fetch_market_context(self, symbol: str) -> dict[str, object]:
        """Fetch market data for enrichment.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with price, volume, market_cap, sector, etc.
        """
        try:
            market_data = await asyncio.to_thread(self.market_fetcher.fetch_daily, symbol, period_days=5)

            latest_row = market_data.data.iloc[-1]
            price = float(latest_row["Close"])
            volume = float(latest_row["Volume"])

            try:
                import yfinance as yf

                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get("marketCap", 0)
                sector = info.get("sector", "Unknown")
                name = info.get("longName") or info.get("shortName", symbol)
            except Exception as e:
                logger.opt(exception=True).debug(f"Failed to fetch yfinance info for {symbol}: {e}")
                market_cap = 0
                sector = "Unknown"
                name = symbol

            return {
                "price": price,
                "volume": volume,
                "market_cap": market_cap,
                "sector": sector,
                "name": name,
            }

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch market context for {symbol}: {e}")
            return {
                "price": 0.0,
                "volume": 0,
                "market_cap": 0,
                "sector": "Unknown",
                "name": symbol,
            }

    def _calculate_event_score(
        self,
        relevance: float,
        urgency: Urgency,
        confidence: float,
        market_data: dict[str, object],
    ) -> float:
        """Calculate composite score from event triage and market data.

        Args:
            relevance: Triage relevance (0-1)
            urgency: Triage urgency
            confidence: Triage confidence (0-1)
            market_data: Market data dict

        Returns:
            Composite score (0-1)
        """
        urgency_boost = {
            Urgency.IMMEDIATE: 0.2,
            Urgency.WATCHLIST: 0.1,
            Urgency.IGNORE: 0.0,
        }

        base_score = (relevance * 0.6) + (confidence * 0.4)

        boost = urgency_boost.get(urgency, 0.0)
        score = min(1.0, base_score + boost)

        volume = market_data.get("volume", 0)
        if isinstance(volume, (int, float)) and volume > 1_000_000:
            score = min(1.0, score + 0.05)

        return score

    def __repr__(self) -> str:
        """Return string representation."""
        return "EventDiscoveryAdapter()"
