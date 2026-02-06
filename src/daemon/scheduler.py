"""Market hours scheduler for the trading daemon."""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from loguru import logger

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

    def get_trading_session(self):  # noqa: ANN201
        """Determine current trading session.

        Returns:
            TradingSession if in session (REGULAR or PRE_MARKET), None if market closed
        """
        from src.strategies.session import TradingSession

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

        # Pre-market check (4:00 AM - 9:30 AM, if enabled)
        if self.enable_pre_market:
            pre_open = time(*PRE_MARKET_START)
            pre_close = time(*PRE_MARKET_END)
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
        """Calculate seconds until market opens (pre-market if enabled, else regular).

        Returns:
            Seconds until market open (0 if already open)
        """
        if self.is_market_open():
            return 0

        now = datetime.now(self.timezone)

        # Determine target open time
        if self.enable_pre_market:
            target_hour, target_minute = PRE_MARKET_START
        else:
            target_hour, target_minute = self.start_hour, self.start_minute

        target_open = now.replace(
            hour=target_hour,
            minute=target_minute,
            second=0,
            microsecond=0,
        )

        # If past today's market close, calculate next trading day
        if now.time() > time(self.end_hour, self.end_minute):
            days_to_add = 1
            if now.weekday() == 4:  # Friday
                days_to_add = 3
            elif now.weekday() == 5:  # Saturday
                days_to_add = 2
            target_open = target_open + timedelta(days=days_to_add)
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

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MarketScheduler({self.start_hour:02d}:{self.start_minute:02d}-"
            f"{self.end_hour:02d}:{self.end_minute:02d} {self.timezone})"
        )
