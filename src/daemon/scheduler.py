"""Market hours scheduler for the trading daemon."""

from datetime import datetime, time
from zoneinfo import ZoneInfo

from loguru import logger


class MarketScheduler:
    """Scheduler that respects market hours."""

    def __init__(
        self,
        start_time: str = "09:30",
        end_time: str = "16:00",
        timezone: str = "America/New_York",
    ) -> None:
        """Initialize market scheduler.

        Args:
            start_time: Market open time (HH:MM format)
            end_time: Market close time (HH:MM format)
            timezone: Market timezone
        """
        self.start_hour, self.start_minute = map(int, start_time.split(":"))
        self.end_hour, self.end_minute = map(int, end_time.split(":"))
        self.timezone = ZoneInfo(timezone)
        logger.info(f"MarketScheduler initialized: {start_time}-{end_time} {timezone}")

    def is_market_open(self) -> bool:
        """Check if market is currently open.

        Returns:
            True if current time is within market hours
        """
        now = datetime.now(self.timezone)
        current_time = now.time()

        market_open = time(self.start_hour, self.start_minute)
        market_close = time(self.end_hour, self.end_minute)

        weekday = now.weekday()
        if weekday >= 5:
            return False

        return market_open <= current_time <= market_close

    def time_until_open(self) -> int:
        """Calculate seconds until market opens.

        Returns:
            Seconds until market open (0 if already open)
        """
        if self.is_market_open():
            return 0

        now = datetime.now(self.timezone)
        market_open = now.replace(
            hour=self.start_hour,
            minute=self.start_minute,
            second=0,
            microsecond=0,
        )

        if now.time() > time(self.end_hour, self.end_minute):
            days_to_add = 1
            if now.weekday() == 4:
                days_to_add = 3
            elif now.weekday() == 5:
                days_to_add = 2
            market_open = market_open.replace(day=now.day + days_to_add)
        elif now.weekday() >= 5:
            days_until_monday = 7 - now.weekday()
            market_open = market_open.replace(day=now.day + days_until_monday)

        return max(0, int((market_open - now).total_seconds()))

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

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MarketScheduler({self.start_hour:02d}:{self.start_minute:02d}-"
            f"{self.end_hour:02d}:{self.end_minute:02d} {self.timezone})"
        )
