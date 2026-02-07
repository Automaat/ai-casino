"""Market hours scheduler for the trading daemon."""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from loguru import logger

from src.strategies.session import TradingSession

PRE_MARKET_START = (4, 0)  # 4:00 AM ET
PRE_MARKET_END = (9, 30)  # 9:30 AM ET (regular market open)
AFTER_HOURS_START = (16, 0)  # 4:00 PM ET (regular market close)
AFTER_HOURS_END = (20, 0)  # 8:00 PM ET


class MarketScheduler:
    """Scheduler that respects market hours."""

    def __init__(  # noqa: PLR0913
        self,
        start_time: str = "09:30",
        end_time: str = "16:00",
        timezone: str = "America/New_York",
        enable_pre_market: bool = False,
        enable_after_hours: bool = False,
        after_hours_screen_time: str = "16:30",
        after_hours_screen_days: list[str] | None = None,
        optimization_time: str = "17:00",
        optimization_days: list[str] | None = None,
    ) -> None:
        """Initialize market scheduler.

        Args:
            start_time: Market open time (HH:MM format)
            end_time: Market close time (HH:MM format)
            timezone: Market timezone
            enable_pre_market: Enable pre-market hours (4:00-9:30 AM ET)
            enable_after_hours: Enable after-hours screening (16:00-20:00 ET)
            after_hours_screen_time: Time to run after-hours screening (HH:MM format)
            after_hours_screen_days: Days to run screening (e.g., ["mon", "tue", "wed", "thu", "fri"])
            optimization_time: Time to run parameter optimization (HH:MM format)
            optimization_days: Days to run optimization (e.g., ["sat"])
        """
        self.start_hour, self.start_minute = map(int, start_time.split(":"))
        self.end_hour, self.end_minute = map(int, end_time.split(":"))
        self.timezone = ZoneInfo(timezone)
        self.enable_pre_market = enable_pre_market
        self.enable_after_hours = enable_after_hours
        self.after_hours_screen_time = after_hours_screen_time
        self.after_hours_screen_days = after_hours_screen_days or ["mon", "tue", "wed", "thu", "fri"]
        self.optimization_time = optimization_time
        self.optimization_days = optimization_days or ["sat"]
        logger.info(
            f"MarketScheduler initialized: {start_time}-{end_time} {timezone} "
            f"(pre-market={'enabled' if enable_pre_market else 'disabled'}, "
            f"after-hours={'enabled' if enable_after_hours else 'disabled'})"
        )

    def get_trading_session(self) -> TradingSession | None:
        """Determine current trading session.

        Returns:
            TradingSession if in session (REGULAR, PRE_MARKET, or AFTER_HOURS), None if market closed
        """
        now = datetime.now(self.timezone)
        current_time = now.time()

        # Weekend check
        if now.weekday() >= 5:
            return None

        # Regular hours check (9:30 AM - 4:00 PM)
        market_open = time(self.start_hour, self.start_minute)
        market_close = time(self.end_hour, self.end_minute)
        if market_open <= current_time < market_close:
            return TradingSession.REGULAR

        # Pre-market check (4:00 AM - configured start_time, if enabled)
        if self.enable_pre_market:
            pre_open = time(*PRE_MARKET_START)
            pre_close = market_open  # Use configured regular market open
            if pre_open <= current_time < pre_close:
                return TradingSession.PRE_MARKET

        # After-hours check (16:00 - 20:00, if enabled)
        if self.enable_after_hours:
            after_open = time(*AFTER_HOURS_START)
            after_close = time(*AFTER_HOURS_END)
            if after_open <= current_time <= after_close:
                return TradingSession.AFTER_HOURS

        return None

    def is_market_open(self) -> bool:
        """Check if market is currently open (regular OR pre-market if enabled).

        AFTER_HOURS is NOT considered "market open" to prevent normal analysis cycles.

        Returns:
            True if current time is within trading hours (regular or pre-market)
        """
        session = self.get_trading_session()
        return session in (TradingSession.REGULAR, TradingSession.PRE_MARKET)

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

        # Use datetime arithmetic to avoid time overflow and handle seconds correctly
        market_close_dt = now.replace(hour=self.end_hour, minute=self.end_minute, second=0, microsecond=0)
        window_start_dt = market_close_dt + timedelta(minutes=offset_minutes)
        window_end_dt = window_start_dt + timedelta(minutes=30)

        return window_start_dt <= now <= window_end_dt

    def is_after_hours_screening_time(self) -> bool:
        """Check if current time matches after-hours screening schedule.

        Returns:
            True if current time is within 1 minute of configured screening time on configured day
        """
        if not self.enable_after_hours:
            return False

        now = datetime.now(self.timezone)

        # Check if current day is in configured days
        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]
        if current_day not in self.after_hours_screen_days:
            return False

        # Parse target time
        target_hour, target_minute = map(int, self.after_hours_screen_time.split(":"))

        # Check if within 1 minute of target time
        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_health_check_time(self, health_run_time: str = "17:00") -> bool:
        """Check if current time matches health check schedule.

        Args:
            health_run_time: Time to run health checks (HH:MM format)

        Returns:
            True if current time is within 1 minute of configured health check time on weekday
        """
        now = datetime.now(self.timezone)

        # Weekend check
        if now.weekday() >= 5:
            return False

        target_hour, target_minute = map(int, health_run_time.split(":"))

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_optimization_time(self) -> bool:
        """Check if current time matches optimization schedule.

        Returns:
            True if current time is within 1 minute of configured optimization time on configured day
        """
        now = datetime.now(self.timezone)

        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]
        if current_day not in self.optimization_days:
            return False

        target_hour, target_minute = map(int, self.optimization_time.split(":"))

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MarketScheduler({self.start_hour:02d}:{self.start_minute:02d}-"
            f"{self.end_hour:02d}:{self.end_minute:02d} {self.timezone})"
        )
