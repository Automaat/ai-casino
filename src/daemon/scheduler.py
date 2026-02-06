"""Market hours scheduler for the trading daemon."""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from loguru import logger

from src.strategies.session import TradingSession

PRE_MARKET_START = (4, 0)  # 4:00 AM ET
PRE_MARKET_END = (9, 30)  # 9:30 AM ET (regular market open)


class MarketScheduler:
    """Scheduler that respects market hours."""

    def __init__(
        self,
        start_time: str = "09:30",
        end_time: str = "16:00",
        timezone: str = "America/New_York",
        enable_pre_market: bool = False,
    ) -> None:
        """Initialize market scheduler.

        Args:
            start_time: Market open time (HH:MM format)
            end_time: Market close time (HH:MM format)
            timezone: Market timezone
            enable_pre_market: Enable pre-market hours (4:00-9:30 AM ET)
        """
        self.start_hour, self.start_minute = map(int, start_time.split(":"))
        self.end_hour, self.end_minute = map(int, end_time.split(":"))
        self.timezone = ZoneInfo(timezone)
        self.enable_pre_market = enable_pre_market
        logger.info(
            f"MarketScheduler initialized: {start_time}-{end_time} {timezone} "
            f"(pre-market={'enabled' if enable_pre_market else 'disabled'})"
        )

    def get_trading_session(self) -> TradingSession | None:
        """Determine current trading session.

        Returns:
            TradingSession if in session (REGULAR or PRE_MARKET), None if market closed
        """
        now = datetime.now(self.timezone)
        current_time = now.time()

        # Weekend check
        if now.weekday() >= 5:
            return None

        # Regular hours check (9:30 AM - 4:00 PM)
        market_open = time(self.start_hour, self.start_minute)
        market_close = time(self.end_hour, self.end_minute)
        if market_open <= current_time <= market_close:
            return TradingSession.REGULAR

        # Pre-market check (4:00 AM - configured start_time, if enabled)
        if self.enable_pre_market:
            pre_open = time(*PRE_MARKET_START)
            pre_close = market_open  # Use configured regular market open
            if pre_open <= current_time < pre_close:
                return TradingSession.PRE_MARKET

        return None

    def is_market_open(self) -> bool:
        """Check if market is currently open (regular OR pre-market if enabled).

        Returns:
            True if current time is within trading hours (regular or pre-market)
        """
        return self.get_trading_session() is not None

    def time_until_open(self) -> int:
        """Calculate seconds until next market open (pre-market if enabled, else regular).

        Returns:
            Seconds until next market open (0 if already open)
        """
        if self.is_market_open():
            return 0

        now = datetime.now(self.timezone)
        current_time = now.time()

        # Determine next target open time
        if self.enable_pre_market:
            pre_open_today = now.replace(
                hour=PRE_MARKET_START[0],
                minute=PRE_MARKET_START[1],
                second=0,
                microsecond=0,
            )
            market_open_today = now.replace(
                hour=self.start_hour,
                minute=self.start_minute,
                second=0,
                microsecond=0,
            )

            # If before pre-market, target pre-market
            if current_time < time(*PRE_MARKET_START):
                target_open = pre_open_today
            # If in gap between pre-market close and regular open, target regular
            elif current_time < time(self.start_hour, self.start_minute):
                target_open = market_open_today
            # Otherwise past regular open, target next day
            else:
                target_open = pre_open_today + timedelta(days=1)
        else:
            # No pre-market, target regular market open
            target_open = now.replace(
                hour=self.start_hour,
                minute=self.start_minute,
                second=0,
                microsecond=0,
            )
            # If past regular open, target next day
            if current_time >= time(self.start_hour, self.start_minute):
                target_open = target_open + timedelta(days=1)

        # If past today's market close, calculate next trading day
        if now.time() > time(self.end_hour, self.end_minute):
            if now.weekday() == 4:  # Friday
                target_open = target_open + timedelta(days=2 if target_open.date() == now.date() else 0)
            elif now.weekday() == 5:  # Saturday
                target_open = target_open + timedelta(days=1)
        elif now.weekday() >= 5:  # Weekend
            days_until_monday = 7 - now.weekday()
            target_open = target_open + timedelta(days=days_until_monday)

        return max(0, int((target_open - now).total_seconds()))

    def time_until_close(self) -> int:
        """Calculate seconds until market closes.

        Returns:
            Seconds until market close (0 if already closed)
        """
        if not self.is_market_open():
            return 0

        now = datetime.now(self.timezone)
        market_close = now.replace(
            hour=self.end_hour,
            minute=self.end_minute,
            second=0,
            microsecond=0,
        )

        return max(0, int((market_close - now).total_seconds()))

    def is_journal_window(self, offset_minutes: int = 15) -> bool:
        """Check if current time is in the after-hours journal window.

        Window starts at market close + offset_minutes, lasts 30 minutes.
        Only active on weekdays.

        Args:
            offset_minutes: Minutes after market close to start journal window

        Returns:
            True if in journal window
        """
        now = datetime.now(self.timezone)

        # Weekend check
        if now.weekday() >= 5:
            return False

        current_time = now.time()
        close_minutes = self.end_hour * 60 + self.end_minute
        window_start_minutes = close_minutes + offset_minutes
        window_end_minutes = window_start_minutes + 30

        window_start = time(window_start_minutes // 60, window_start_minutes % 60)
        window_end = time(window_end_minutes // 60, window_end_minutes % 60)

        return window_start <= current_time <= window_end

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MarketScheduler({self.start_hour:02d}:{self.start_minute:02d}-"
            f"{self.end_hour:02d}:{self.end_minute:02d} {self.timezone})"
        )
