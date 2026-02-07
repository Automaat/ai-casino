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
        health_check_time: str = "17:00",
        prefetch_time: str = "16:30",
        pre_market_refresh_time: str = "04:00",
        sector_rotation_time: str = "16:15",
        sector_rotation_days: list[str] | None = None,
        enable_sector_rotation: bool = False,
        earnings_fetch_time: str = "16:45",
        earnings_fetch_days: list[str] | None = None,
        enable_earnings_calendar: bool = False,
        peer_analysis_time: str = "17:30",
        peer_analysis_days: list[str] | None = None,
        enable_peer_analysis: bool = False,
        correlation_audit_time: str = "17:45",
        correlation_audit_days: list[str] | None = None,
        enable_correlation_audit: bool = False,
        tearsheet_time: str = "16:30",
        enable_reporting: bool = False,
        rebalancing_time: str = "16:45",
        rebalancing_days: list[str] | None = None,
        enable_rebalancing: bool = False,
        signal_tracking_time: str = "17:00",
        enable_signal_tracking: bool = True,
        game_plan_time: str = "04:00",
        enable_game_plan: bool = False,
        monte_carlo_time: str = "17:00",
        monte_carlo_days: list[str] | None = None,
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
            health_check_time: Time to run health check (HH:MM format)
            prefetch_time: Time to run after-hours data prefetch (HH:MM format)
            pre_market_refresh_time: Time to run pre-market data refresh (HH:MM format)
            sector_rotation_time: Time to run sector rotation analysis (HH:MM format)
            sector_rotation_days: Days to run sector rotation (e.g., ["mon", ..., "fri"])
            enable_sector_rotation: Enable sector rotation analysis
            earnings_fetch_time: Time to fetch earnings calendar (HH:MM format)
            earnings_fetch_days: Days to fetch earnings (e.g., ["mon"])
            enable_earnings_calendar: Enable earnings calendar fetching
            peer_analysis_time: Time to run deep peer analysis (HH:MM format)
            peer_analysis_days: Days to run peer analysis (e.g., ["sun"])
            enable_peer_analysis: Enable deep peer analysis
            correlation_audit_time: Time to run correlation audit (HH:MM format)
            correlation_audit_days: Days to run correlation audit (e.g., ["sun"])
            enable_correlation_audit: Enable correlation audit
            tearsheet_time: Time to generate tearsheets (HH:MM format)
            enable_reporting: Enable tearsheet generation
            rebalancing_time: Time to run portfolio rebalancing (HH:MM format)
            rebalancing_days: Days to run rebalancing (e.g., ["mon"])
            enable_rebalancing: Enable portfolio rebalancing
            signal_tracking_time: Time to run signal tracking (HH:MM format)
            enable_signal_tracking: Enable signal tracking
            game_plan_time: Time to generate game plan (HH:MM format)
            enable_game_plan: Enable game plan generation
            monte_carlo_time: Time to run Monte Carlo stress test (HH:MM format)
            monte_carlo_days: Days to run stress test (e.g., ["sun"])
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
        self.health_check_time = health_check_time
        self.prefetch_time = prefetch_time
        self.pre_market_refresh_time = pre_market_refresh_time
        self.sector_rotation_time = sector_rotation_time
        self.sector_rotation_days = sector_rotation_days or ["mon", "tue", "wed", "thu", "fri"]
        self.enable_sector_rotation = enable_sector_rotation
        self.earnings_fetch_time = earnings_fetch_time
        self.earnings_fetch_days = earnings_fetch_days or ["mon"]
        self.enable_earnings_calendar = enable_earnings_calendar
        self.peer_analysis_time = peer_analysis_time
        self.peer_analysis_days = peer_analysis_days or ["sun"]
        self.enable_peer_analysis = enable_peer_analysis
        self.correlation_audit_time = correlation_audit_time
        self.correlation_audit_days = correlation_audit_days or ["sun"]
        self.enable_correlation_audit = enable_correlation_audit
        self.tearsheet_time = tearsheet_time
        self.enable_reporting = enable_reporting
        self.rebalancing_time = rebalancing_time
        self.rebalancing_days = rebalancing_days or ["mon"]
        self.enable_rebalancing = enable_rebalancing
        self.signal_tracking_time = signal_tracking_time
        self.enable_signal_tracking = enable_signal_tracking
        self.game_plan_time = game_plan_time
        self.enable_game_plan = enable_game_plan
        self.monte_carlo_time = monte_carlo_time
        self.monte_carlo_days = monte_carlo_days or ["sun"]
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

    def is_health_check_time(self, health_run_time: str | None = None) -> bool:
        """Check if current time matches health check schedule.

        Args:
            health_run_time: Time to run health checks (HH:MM format). If None, uses self.health_check_time.

        Returns:
            True if current time is within 1 minute of configured health check time on weekday
        """
        now = datetime.now(self.timezone)

        # Weekend check
        if now.weekday() >= 5:
            return False

        run_time = health_run_time if health_run_time is not None else self.health_check_time
        try:
            target_hour, target_minute = map(int, run_time.split(":"))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Malformed health_run_time '{run_time}': {e}")
            return False

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

    def is_portfolio_rebalancing_time(self) -> bool:
        """Check if current time matches portfolio rebalancing schedule.

        Returns:
            True if current time is within 1 minute of configured rebalancing time on configured day
        """
        if not self.enable_rebalancing:
            return False

        now = datetime.now(self.timezone)

        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]
        if current_day not in self.rebalancing_days:
            return False

        target_hour, target_minute = map(int, self.rebalancing_time.split(":"))

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_prefetch_time(self) -> bool:
        """Check if current time matches after-hours prefetch schedule.

        Returns:
            True if current time is within 1 minute of configured prefetch time on weekday
        """
        now = datetime.now(self.timezone)

        # Weekday check
        if now.weekday() >= 5:
            return False

        try:
            target_hour, target_minute = map(int, self.prefetch_time.split(":"))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Malformed prefetch_time '{self.prefetch_time}': {e}")
            return False

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_pre_market_refresh_time(self) -> bool:
        """Check if current time matches pre-market refresh schedule.

        Returns:
            True if current time is within 1 minute of configured refresh time on weekday
        """
        now = datetime.now(self.timezone)

        # Weekday check
        if now.weekday() >= 5:
            return False

        try:
            target_hour, target_minute = map(int, self.pre_market_refresh_time.split(":"))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Malformed pre_market_refresh_time '{self.pre_market_refresh_time}': {e}")
            return False

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_sector_rotation_time(self) -> bool:
        """Check if current time matches sector rotation schedule.

        Returns:
            True if within 1 minute of configured time on configured day
        """
        if not self.enable_sector_rotation:
            return False

        now = datetime.now(self.timezone)

        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]
        if current_day not in self.sector_rotation_days:
            return False

        target_hour, target_minute = map(int, self.sector_rotation_time.split(":"))

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_earnings_fetch_time(self) -> bool:
        """Check if current time matches earnings calendar fetch schedule.

        Returns:
            True if within 1 minute of configured time on configured day
        """
        if not self.enable_earnings_calendar:
            return False

        now = datetime.now(self.timezone)

        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]
        if current_day not in self.earnings_fetch_days:
            return False

        target_hour, target_minute = map(int, self.earnings_fetch_time.split(":"))

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_peer_analysis_time(self) -> bool:
        """Check if current time matches peer analysis schedule.

        Returns:
            True if within 1 minute of configured time on configured day
        """
        if not self.enable_peer_analysis:
            return False

        now = datetime.now(self.timezone)

        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]
        if current_day not in self.peer_analysis_days:
            return False

        target_hour, target_minute = map(int, self.peer_analysis_time.split(":"))

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_correlation_audit_time(self) -> bool:
        """Check if current time matches correlation audit schedule.

        Returns:
            True if within 1 minute of configured time on configured day
        """
        if not self.enable_correlation_audit:
            return False

        now = datetime.now(self.timezone)

        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]
        if current_day not in self.correlation_audit_days:
            return False

        target_hour, target_minute = map(int, self.correlation_audit_time.split(":"))

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_tearsheet_time(self) -> bool:
        """Check if current time matches tearsheet generation schedule.

        Returns:
            True if within 1 minute of configured time on weekday
        """
        if not self.enable_reporting:
            return False

        now = datetime.now(self.timezone)

        # Weekday check
        if now.weekday() >= 5:
            return False

        try:
            target_hour, target_minute = map(int, self.tearsheet_time.split(":"))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Malformed tearsheet_time '{self.tearsheet_time}': {e}")
            return False

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_signal_tracking_time(self) -> bool:
        """Check if current time matches signal tracking schedule.

        Returns:
            True if within 1 minute of configured time on weekday
        """
        if not self.enable_signal_tracking:
            return False

        now = datetime.now(self.timezone)

        # Weekday check
        if now.weekday() >= 5:
            return False

        try:
            target_hour, target_minute = map(int, self.signal_tracking_time.split(":"))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Malformed signal_tracking_time '{self.signal_tracking_time}': {e}")
            return False

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_rebalancing_time(self) -> bool:
        """Check if current time matches portfolio rebalancing schedule.

        Returns:
            True if within 1 minute of configured time on configured day
        """
        if not self.enable_rebalancing:
            return False

        now = datetime.now(self.timezone)

        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]
        if current_day not in self.rebalancing_days:
            return False

        try:
            target_hour, target_minute = map(int, self.rebalancing_time.split(":"))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Malformed rebalancing_time '{self.rebalancing_time}': {e}")
            return False

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_game_plan_time(self) -> bool:
        """Check if it's time to generate game plan (04:00 ET, weekdays only).

        Returns:
            True if within 1 minute of configured time on weekday
        """
        if not self.enable_game_plan:
            return False

        now = datetime.now(self.timezone)

        if now.weekday() >= 5:
            return False

        try:
            target_hour, target_minute = map(int, self.game_plan_time.split(":"))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Malformed game_plan_time '{self.game_plan_time}': {e}")
            return False

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_monte_carlo_time(self) -> bool:
        """Check if current time matches Monte Carlo stress test schedule.

        Returns:
            True if current time is within 1 minute of configured time on configured day
        """
        now = datetime.now(self.timezone)

        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]
        if current_day not in self.monte_carlo_days:
            return False

        target_hour, target_minute = map(int, self.monte_carlo_time.split(":"))

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MarketScheduler({self.start_hour:02d}:{self.start_minute:02d}-"
            f"{self.end_hour:02d}:{self.end_minute:02d} {self.timezone})"
        )
