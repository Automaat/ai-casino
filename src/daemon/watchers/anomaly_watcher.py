"""Anomaly watcher for market data anomalies (volume spikes, price moves, gaps).

Polls Alpha Vantage intraday data every 15 minutes, uses round-robin rotation through watchlist,
maintains volume baselines and previous close cache, detects multiple anomaly types per symbol.
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.event_watcher import EventWatcher, EventWatcherConfig
from src.daemon.events import AnomalyEvent, BaseEvent, Gap, PriceMove, VolumeSpike
from src.data.market import MarketDataFetcher


@dataclass
class AnomalyWatcherConfig:
    """Configuration for AnomalyWatcher."""

    poll_interval: int = 900
    relevance_threshold: float = 0.7
    cooldown_minutes: int = 15
    volume_spike_multiplier: float = 2.0
    price_move_threshold_pct: float = 5.0
    gap_threshold_pct: float = 3.0
    watchlist: list[str] = field(default_factory=list)
    max_symbols_per_cycle: int = 5
    max_concurrent_analyses: int = 2


class AnomalyWatcher(EventWatcher):
    """Watcher for market data anomalies.

    Monitors watchlist for volume spikes, large intraday moves, and gaps.
    Uses round-robin rotation to check full watchlist over multiple polls.
    """

    def __init__(  # noqa: PLR0913,D417 - Backward compat, prefer AnomalyWatcherConfig
        self,
        historical_cache: HistoricalCache,
        market_fetcher: MarketDataFetcher,
        config: AnomalyWatcherConfig | None = None,
        poll_interval: int | None = None,
        relevance_threshold: float | None = None,
        cooldown_minutes: int | None = None,
        volume_spike_multiplier: float | None = None,
        price_move_threshold_pct: float | None = None,
        gap_threshold_pct: float | None = None,
        watchlist: list[str] | None = None,
        max_symbols_per_cycle: int | None = None,
        max_concurrent_analyses: int | None = None,
    ) -> None:
        """Initialize anomaly watcher.

        Args:
            historical_cache: Shared cache for market data
            market_fetcher: Market data fetcher for Alpha Vantage
            config: Configuration (uses defaults if not provided)
            **Individual params for backward compatibility (prefer config object)
        """
        # Backward compat: construct config from individual params if provided
        if config is None and (
            poll_interval is not None
            or relevance_threshold is not None
            or cooldown_minutes is not None
            or volume_spike_multiplier is not None
            or price_move_threshold_pct is not None
            or gap_threshold_pct is not None
            or watchlist is not None
            or max_symbols_per_cycle is not None
            or max_concurrent_analyses is not None
        ):
            defaults = AnomalyWatcherConfig()
            config = AnomalyWatcherConfig(
                poll_interval=poll_interval if poll_interval is not None else defaults.poll_interval,
                relevance_threshold=(
                    relevance_threshold if relevance_threshold is not None else defaults.relevance_threshold
                ),
                cooldown_minutes=(
                    cooldown_minutes if cooldown_minutes is not None else defaults.cooldown_minutes
                ),
                volume_spike_multiplier=(
                    volume_spike_multiplier
                    if volume_spike_multiplier is not None
                    else defaults.volume_spike_multiplier
                ),
                price_move_threshold_pct=(
                    price_move_threshold_pct
                    if price_move_threshold_pct is not None
                    else defaults.price_move_threshold_pct
                ),
                gap_threshold_pct=(
                    gap_threshold_pct if gap_threshold_pct is not None else defaults.gap_threshold_pct
                ),
                watchlist=watchlist if watchlist is not None else defaults.watchlist,
                max_symbols_per_cycle=(
                    max_symbols_per_cycle
                    if max_symbols_per_cycle is not None
                    else defaults.max_symbols_per_cycle
                ),
                max_concurrent_analyses=(
                    max_concurrent_analyses
                    if max_concurrent_analyses is not None
                    else defaults.max_concurrent_analyses
                ),
            )

        cfg = config or AnomalyWatcherConfig()
        base_config = EventWatcherConfig(
            poll_interval=cfg.poll_interval,
            relevance_threshold=cfg.relevance_threshold,
            cooldown_minutes=cfg.cooldown_minutes,
            max_concurrent_analyses=cfg.max_concurrent_analyses,
        )
        super().__init__(base_config, historical_cache)
        self._market_fetcher = market_fetcher
        self.volume_spike_multiplier = cfg.volume_spike_multiplier
        self.price_move_threshold_pct = cfg.price_move_threshold_pct
        self.gap_threshold_pct = cfg.gap_threshold_pct
        self.watchlist = cfg.watchlist
        self.max_symbols_per_cycle = cfg.max_symbols_per_cycle

        # State tracking
        self._volume_baselines: OrderedDict[str, float] = OrderedDict()  # LRU cache
        self._previous_close_cache: dict[str, float] = {}
        self._last_cache_refresh_date: datetime | None = None
        self._rotation_offset = 0

        logger.info(
            f"AnomalyWatcher initialized (volume_spike={cfg.volume_spike_multiplier}x, "
            f"price_move={cfg.price_move_threshold_pct}%, gap={cfg.gap_threshold_pct}%, "
            f"max_per_cycle={cfg.max_symbols_per_cycle}, watchlist={len(self.watchlist)} symbols)"
        )

    def _init_components(self) -> None:
        """Lazy initialization of parent components."""
        super()._init_components()

    def _get_next_symbols(self) -> list[str]:
        """Get next batch of symbols using round-robin rotation.

        Returns:
            List of symbols to check (wraps around watchlist)
        """
        if not self.watchlist:
            return []

        max_symbols = min(self.max_symbols_per_cycle, len(self.watchlist))
        start = self._rotation_offset
        end = start + max_symbols

        if end <= len(self.watchlist):
            symbols = self.watchlist[start:end]
        else:
            # Wrap around
            symbols = self.watchlist[start:] + self.watchlist[: end - len(self.watchlist)]

        self._rotation_offset = end % len(self.watchlist)
        logger.debug(f"Round-robin: checking {symbols} (offset now {self._rotation_offset})")
        return symbols

    def _update_volume_baseline(self, symbol: str, avg_volume: float) -> None:
        """Update volume baseline with LRU eviction.

        Args:
            symbol: Stock ticker
            avg_volume: 20-day average volume
        """
        if symbol in self._volume_baselines:
            self._volume_baselines.move_to_end(symbol)

        self._volume_baselines[symbol] = avg_volume

        # LRU eviction at 300 symbols
        if len(self._volume_baselines) > 300:
            oldest = next(iter(self._volume_baselines))
            del self._volume_baselines[oldest]
            logger.debug(f"LRU eviction: removed {oldest} baseline")

    def _refresh_previous_close_if_needed(self) -> None:
        """Clear previous close cache on new trading day."""
        now = datetime.now(UTC)
        today = now.date()

        if self._last_cache_refresh_date is None:
            self._last_cache_refresh_date = now
            return

        last_refresh_date = self._last_cache_refresh_date.date()

        # Clear cache if date changed (new day)
        if today > last_refresh_date:
            logger.info("New trading day detected, clearing previous close cache")
            self._previous_close_cache.clear()
            self._last_cache_refresh_date = now

    async def _detect_volume_spike(self, symbol: str, current_volume: float) -> VolumeSpike | None:
        """Detect volume spike for symbol."""
        if self._market_fetcher is None:
            msg = "Market fetcher not initialized"
            raise RuntimeError(msg)
        if symbol not in self._volume_baselines:
            # Establish baseline from daily data
            try:
                daily = await asyncio.to_thread(self._market_fetcher.fetch_daily, symbol, 30)
                if not daily.data.empty:
                    avg_vol = float(daily.data["Volume"].tail(20).mean())
                    self._update_volume_baseline(symbol, avg_vol)
                    logger.debug(f"Established baseline for {symbol}: {avg_vol:,.0f}")
            except Exception as e:
                logger.warning(f"Failed to establish baseline for {symbol}: {e}")
                return None

        if symbol not in self._volume_baselines:
            return None

        avg_vol = self._volume_baselines[symbol]
        if avg_vol <= 0:
            return None

        multiplier = current_volume / avg_vol
        if multiplier >= self.volume_spike_multiplier:
            logger.info(f"Volume spike: {symbol} {multiplier:.1f}x ({current_volume:,.0f} vs {avg_vol:,.0f})")
            return VolumeSpike(
                current_volume=current_volume,
                avg_volume_20d=avg_vol,
                spike_multiplier=multiplier,
            )
        return None

    def _detect_price_move(
        self, symbol: str, open_price: float, current_price: float, high: float, low: float
    ) -> PriceMove | None:
        """Detect intraday price move for symbol."""
        if open_price <= 0:
            return None

        change_pct = ((current_price - open_price) / open_price) * 100
        if abs(change_pct) >= self.price_move_threshold_pct:
            logger.info(f"Price move: {symbol} {change_pct:+.1f}%")
            return PriceMove(
                open_price=open_price,
                current_price=current_price,
                change_pct=change_pct,
                high=high,
                low=low,
            )
        return None

    async def _detect_gap(self, symbol: str, open_price: float) -> Gap | None:
        """Detect gap for symbol."""
        if self._market_fetcher is None:
            msg = "Market fetcher not initialized"
            raise RuntimeError(msg)
        if symbol not in self._previous_close_cache:
            # Fetch prev close from daily data
            try:
                daily = await asyncio.to_thread(self._market_fetcher.fetch_daily, symbol, 2)
                if len(daily.data) >= 2:
                    prev_close = float(daily.data["Close"].iloc[-2])
                    self._previous_close_cache[symbol] = prev_close
                    logger.debug(f"Cached prev close for {symbol}: ${prev_close:.2f}")
            except Exception as e:
                logger.warning(f"Failed to fetch prev close for {symbol}: {e}")
                return None

        if symbol not in self._previous_close_cache:
            return None

        prev_close = self._previous_close_cache[symbol]
        if prev_close <= 0 or open_price <= 0:
            return None

        gap_pct = ((open_price - prev_close) / prev_close) * 100
        if abs(gap_pct) >= self.gap_threshold_pct:
            gap_direction = "up" if gap_pct > 0 else "down"
            logger.info(f"Gap: {symbol} {gap_direction} {abs(gap_pct):.1f}%")
            return Gap(
                previous_close=prev_close,
                open_price=open_price,
                gap_pct=gap_pct,
                gap_direction=gap_direction,
            )
        return None

    async def _process_symbol_for_anomalies(self, symbol: str) -> AnomalyEvent | None:
        """Process single symbol and detect anomalies.

        Args:
            symbol: Symbol to process

        Returns:
            AnomalyEvent if anomalies detected, None otherwise
        """
        if self._market_fetcher is None:
            msg = "Market fetcher not initialized"
            raise RuntimeError(msg)
        # Fetch intraday data
        try:
            intraday = await asyncio.to_thread(self._market_fetcher.fetch_intraday, symbol, "60min")
        except Exception as e:
            logger.warning(f"Failed to fetch intraday for {symbol}: {e}")
            return None

        if intraday.data.empty:
            logger.debug(f"No intraday data for {symbol}")
            return None

        # Aggregate current trading day bars
        latest_ts = intraday.data.index[-1]
        current_date = latest_ts.date()
        index = intraday.data.index
        if hasattr(index, "date"):
            same_day_mask = index.date == current_date
        else:
            same_day_mask = index.map(lambda x: x.date()) == current_date
        day_bars = intraday.data[same_day_mask]

        if day_bars.empty:
            day_bars = intraday.data.iloc[[-1]]

        # Extract day-level aggregated metrics
        open_price = float(day_bars["Open"].iloc[0])
        current_price = float(day_bars["Close"].iloc[-1])
        high = float(day_bars["High"].max())
        low = float(day_bars["Low"].min())
        current_volume = float(day_bars["Volume"].sum())

        # Run anomaly detections
        anomaly_types = []
        volume_spike_data = await self._detect_volume_spike(symbol, current_volume)
        if volume_spike_data:
            anomaly_types.append("volume_spike")

        price_move_data = self._detect_price_move(symbol, open_price, current_price, high, low)
        if price_move_data:
            anomaly_types.append("price_move")

        gap_data = await self._detect_gap(symbol, open_price)
        if gap_data:
            anomaly_types.append("gap")

        # Create event if anomalies detected
        if anomaly_types:
            event_id = f"{symbol}-{datetime.now(UTC).isoformat()}"
            logger.info(f"Anomaly detected: {symbol} ({'+'.join(anomaly_types)})")
            return AnomalyEvent(
                event_id=event_id,
                event_type="anomaly",
                timestamp=datetime.now(UTC),
                source="market_data",
                symbol=symbol,
                anomaly_types=anomaly_types,
                volume_spike_data=volume_spike_data,
                price_move_data=price_move_data,
                gap_data=gap_data,
            )

        return None

    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch anomaly events (volume spikes, price moves, gaps)."""
        self._init_components()
        if self._market_fetcher is None:
            msg = "Market fetcher not initialized"
            raise RuntimeError(msg)
        self._refresh_previous_close_if_needed()

        if not self.watchlist:
            logger.warning("No watchlist configured")
            return []

        symbols_to_check = self._get_next_symbols()
        if not symbols_to_check:
            return []

        from typing import cast

        events: list[BaseEvent] = []
        for symbol in symbols_to_check:
            try:
                event = await self._process_symbol_for_anomalies(symbol)
                if event:
                    events.append(cast("BaseEvent", event))
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        return events

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AnomalyWatcher(poll_interval={self.poll_interval}s, "
            f"watchlist={len(self.watchlist)} symbols, "
            f"max_per_cycle={self.max_symbols_per_cycle})"
        )
