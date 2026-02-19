"""Market awareness service — focused wrapper around MarketScheduler."""

from datetime import datetime, timedelta

from src.daemon.scheduler import MarketScheduler
from src.strategies.session import TradingSession


class MarketService:
    """Focused service for market-state queries, delegating to MarketScheduler."""

    def __init__(self, scheduler: MarketScheduler) -> None:
        """Initialize service.

        Args:
            scheduler: MarketScheduler instance that owns scheduling logic
        """
        self._scheduler = scheduler

    def current_session(self) -> TradingSession | None:
        """Return current trading session or None if market is closed."""
        return self._scheduler.get_trading_session()

    def is_open(self) -> bool:
        """Return True if market is open (REGULAR or PRE_MARKET if enabled)."""
        return self._scheduler.is_market_open()

    def is_regular_session(self) -> bool:
        """Return True if current session is regular market hours (9:30-16:00 ET)."""
        return self._scheduler.get_trading_session() == TradingSession.REGULAR

    def next_regular_open(self) -> datetime:
        """Return tz-aware ET datetime of the next regular session open (09:30).

        If currently in regular session, returns today's open (already passed).
        Otherwise advances to next trading day (skips weekends).
        """
        tz = self._scheduler.timezone
        now = datetime.now(tz)
        target = now.replace(
            hour=self._scheduler.start_hour,
            minute=self._scheduler.start_minute,
            second=0,
            microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        # Skip weekends
        while target.weekday() >= 5:
            target += timedelta(days=1)
        return target

    def time_until_open(self) -> int:
        """Return seconds until next market open (0 if already open)."""
        return self._scheduler.time_until_open()

    def time_until_close(self) -> int:
        """Return seconds until market close (0 if already closed)."""
        return self._scheduler.time_until_close()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MarketService(open={self._scheduler.start_hour:02d}:{self._scheduler.start_minute:02d}, "
            f"tz={self._scheduler.timezone.key})"
        )
