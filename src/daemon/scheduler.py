"""Market hours scheduler for the trading daemon."""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from loguru import logger

from src.strategies.session import TradingSession

PRE_MARKET_START = (4, 0)  # 4:00 AM ET
PRE_MARKET_END = (9, 30)  # 9:30 AM ET (regular market open)
AFTER_HOURS_START = (16, 0)  # 4:00 PM ET (regular market close)
AFTER_HOURS_END = (20, 0)  # 8:00 PM ET


@dataclass
class MarketSchedulerConfig:
    """Configuration for MarketScheduler."""

    start_time: str = "09:30"
    end_time: str = "16:00"
    timezone: str = "America/New_York"
    enable_pre_market: bool = False
    enable_after_hours: bool = False
    optimization_time: str = "17:00"
    optimization_days: list[str] | None = None
    prefetch_time: str = "16:30"
    pre_market_refresh_time: str = "04:00"
    sector_rotation_time: str = "16:15"
    sector_rotation_days: list[str] | None = None
    enable_sector_rotation: bool = False
    earnings_fetch_time: str = "16:45"
    earnings_fetch_days: list[str] | None = None
    enable_earnings_calendar: bool = False
    peer_analysis_time: str = "17:30"
    peer_analysis_days: list[str] | None = None
    enable_peer_analysis: bool = False
    correlation_audit_time: str = "17:45"
    correlation_audit_days: list[str] | None = None
    enable_correlation_audit: bool = False
    tearsheet_time: str = "16:30"
    enable_reporting: bool = False
    rebalancing_time: str = "16:45"
    rebalancing_days: list[str] | None = None
    enable_rebalancing: bool = False
    signal_tracking_time: str = "17:00"
    enable_signal_tracking: bool = True
    game_plan_time: str = "04:00"
    enable_game_plan: bool = False
    pre_market_screening_time: str = "07:00"
    enable_pre_market_screening: bool = False
    monte_carlo_time: str = "17:00"
    monte_carlo_days: list[str] | None = None
    discovery_outcome_time: str = "17:15"
    discovery_outcome_days: list[str] | None = None
    enable_discovery_outcome: bool = True


class MarketScheduler:
    """Scheduler that respects market hours."""

    def __init__(  # noqa: PLR0913,D417 - Backward compat with tests, prefer MarketSchedulerConfig
        self,
        config: MarketSchedulerConfig | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        timezone: str | None = None,
        enable_pre_market: bool | None = None,
        enable_after_hours: bool | None = None,
        optimization_time: str | None = None,
        optimization_days: list[str] | None = None,
        prefetch_time: str | None = None,
        pre_market_refresh_time: str | None = None,
        sector_rotation_time: str | None = None,
        sector_rotation_days: list[str] | None = None,
        enable_sector_rotation: bool | None = None,
        earnings_fetch_time: str | None = None,
        earnings_fetch_days: list[str] | None = None,
        enable_earnings_calendar: bool | None = None,
        peer_analysis_time: str | None = None,
        peer_analysis_days: list[str] | None = None,
        enable_peer_analysis: bool | None = None,
        correlation_audit_time: str | None = None,
        correlation_audit_days: list[str] | None = None,
        enable_correlation_audit: bool | None = None,
        tearsheet_time: str | None = None,
        enable_reporting: bool | None = None,
        rebalancing_time: str | None = None,
        rebalancing_days: list[str] | None = None,
        enable_rebalancing: bool | None = None,
        signal_tracking_time: str | None = None,
        enable_signal_tracking: bool | None = None,
        game_plan_time: str | None = None,
        enable_game_plan: bool | None = None,
        pre_market_screening_time: str | None = None,
        enable_pre_market_screening: bool | None = None,
        monte_carlo_time: str | None = None,
        monte_carlo_days: list[str] | None = None,
        discovery_outcome_time: str | None = None,
        discovery_outcome_days: list[str] | None = None,
        enable_discovery_outcome: bool | None = None,
    ) -> None:
        """Initialize market scheduler.

        Args:
            config: Configuration (uses defaults if not provided)
            **Individual params for backward compatibility (prefer config object)
        """
        # Backward compat: construct config from individual params if provided
        if config is None and (
            start_time is not None
            or end_time is not None
            or timezone is not None
            or enable_pre_market is not None
            or enable_after_hours is not None
            or optimization_time is not None
            or optimization_days is not None
            or prefetch_time is not None
            or pre_market_refresh_time is not None
            or sector_rotation_time is not None
            or sector_rotation_days is not None
            or enable_sector_rotation is not None
            or earnings_fetch_time is not None
            or earnings_fetch_days is not None
            or enable_earnings_calendar is not None
            or peer_analysis_time is not None
            or peer_analysis_days is not None
            or enable_peer_analysis is not None
            or correlation_audit_time is not None
            or correlation_audit_days is not None
            or enable_correlation_audit is not None
            or tearsheet_time is not None
            or enable_reporting is not None
            or rebalancing_time is not None
            or rebalancing_days is not None
            or enable_rebalancing is not None
            or signal_tracking_time is not None
            or enable_signal_tracking is not None
            or game_plan_time is not None
            or enable_game_plan is not None
            or pre_market_screening_time is not None
            or enable_pre_market_screening is not None
            or monte_carlo_time is not None
            or monte_carlo_days is not None
            or discovery_outcome_time is not None
            or discovery_outcome_days is not None
            or enable_discovery_outcome is not None
        ):
            # Construct config from individual params (overriding defaults)
            defaults = MarketSchedulerConfig()
            config = MarketSchedulerConfig(
                start_time=start_time if start_time is not None else defaults.start_time,
                end_time=end_time if end_time is not None else defaults.end_time,
                timezone=timezone if timezone is not None else defaults.timezone,
                enable_pre_market=(
                    enable_pre_market if enable_pre_market is not None else defaults.enable_pre_market
                ),
                enable_after_hours=(
                    enable_after_hours if enable_after_hours is not None else defaults.enable_after_hours
                ),
                optimization_time=(
                    optimization_time if optimization_time is not None else defaults.optimization_time
                ),
                optimization_days=(
                    optimization_days if optimization_days is not None else defaults.optimization_days
                ),
                prefetch_time=prefetch_time if prefetch_time is not None else defaults.prefetch_time,
                pre_market_refresh_time=(
                    pre_market_refresh_time
                    if pre_market_refresh_time is not None
                    else defaults.pre_market_refresh_time
                ),
                sector_rotation_time=(
                    sector_rotation_time
                    if sector_rotation_time is not None
                    else defaults.sector_rotation_time
                ),
                sector_rotation_days=(
                    sector_rotation_days
                    if sector_rotation_days is not None
                    else defaults.sector_rotation_days
                ),
                enable_sector_rotation=(
                    enable_sector_rotation
                    if enable_sector_rotation is not None
                    else defaults.enable_sector_rotation
                ),
                earnings_fetch_time=(
                    earnings_fetch_time if earnings_fetch_time is not None else defaults.earnings_fetch_time
                ),
                earnings_fetch_days=(
                    earnings_fetch_days if earnings_fetch_days is not None else defaults.earnings_fetch_days
                ),
                enable_earnings_calendar=(
                    enable_earnings_calendar
                    if enable_earnings_calendar is not None
                    else defaults.enable_earnings_calendar
                ),
                peer_analysis_time=(
                    peer_analysis_time if peer_analysis_time is not None else defaults.peer_analysis_time
                ),
                peer_analysis_days=(
                    peer_analysis_days if peer_analysis_days is not None else defaults.peer_analysis_days
                ),
                enable_peer_analysis=(
                    enable_peer_analysis
                    if enable_peer_analysis is not None
                    else defaults.enable_peer_analysis
                ),
                correlation_audit_time=(
                    correlation_audit_time
                    if correlation_audit_time is not None
                    else defaults.correlation_audit_time
                ),
                correlation_audit_days=(
                    correlation_audit_days
                    if correlation_audit_days is not None
                    else defaults.correlation_audit_days
                ),
                enable_correlation_audit=(
                    enable_correlation_audit
                    if enable_correlation_audit is not None
                    else defaults.enable_correlation_audit
                ),
                tearsheet_time=tearsheet_time if tearsheet_time is not None else defaults.tearsheet_time,
                enable_reporting=(
                    enable_reporting if enable_reporting is not None else defaults.enable_reporting
                ),
                rebalancing_time=(
                    rebalancing_time if rebalancing_time is not None else defaults.rebalancing_time
                ),
                rebalancing_days=(
                    rebalancing_days if rebalancing_days is not None else defaults.rebalancing_days
                ),
                enable_rebalancing=(
                    enable_rebalancing if enable_rebalancing is not None else defaults.enable_rebalancing
                ),
                signal_tracking_time=(
                    signal_tracking_time
                    if signal_tracking_time is not None
                    else defaults.signal_tracking_time
                ),
                enable_signal_tracking=(
                    enable_signal_tracking
                    if enable_signal_tracking is not None
                    else defaults.enable_signal_tracking
                ),
                game_plan_time=game_plan_time if game_plan_time is not None else defaults.game_plan_time,
                enable_game_plan=(
                    enable_game_plan if enable_game_plan is not None else defaults.enable_game_plan
                ),
                pre_market_screening_time=(
                    pre_market_screening_time
                    if pre_market_screening_time is not None
                    else defaults.pre_market_screening_time
                ),
                enable_pre_market_screening=(
                    enable_pre_market_screening
                    if enable_pre_market_screening is not None
                    else defaults.enable_pre_market_screening
                ),
                monte_carlo_time=(
                    monte_carlo_time if monte_carlo_time is not None else defaults.monte_carlo_time
                ),
                monte_carlo_days=(
                    monte_carlo_days if monte_carlo_days is not None else defaults.monte_carlo_days
                ),
                discovery_outcome_time=(
                    discovery_outcome_time
                    if discovery_outcome_time is not None
                    else defaults.discovery_outcome_time
                ),
                discovery_outcome_days=(
                    discovery_outcome_days
                    if discovery_outcome_days is not None
                    else defaults.discovery_outcome_days
                ),
                enable_discovery_outcome=(
                    enable_discovery_outcome
                    if enable_discovery_outcome is not None
                    else defaults.enable_discovery_outcome
                ),
            )

        cfg = config or MarketSchedulerConfig()
        self.start_hour, self.start_minute = map(int, cfg.start_time.split(":"))
        self.end_hour, self.end_minute = map(int, cfg.end_time.split(":"))
        self.timezone = ZoneInfo(cfg.timezone)
        self.enable_pre_market = cfg.enable_pre_market
        self.enable_after_hours = cfg.enable_after_hours
        self.optimization_time = cfg.optimization_time
        self.optimization_days = cfg.optimization_days or ["sat"]
        self.prefetch_time = cfg.prefetch_time
        self.pre_market_refresh_time = cfg.pre_market_refresh_time
        self.sector_rotation_time = cfg.sector_rotation_time
        self.sector_rotation_days = cfg.sector_rotation_days or ["mon", "tue", "wed", "thu", "fri"]
        self.enable_sector_rotation = cfg.enable_sector_rotation
        self.earnings_fetch_time = cfg.earnings_fetch_time
        self.earnings_fetch_days = cfg.earnings_fetch_days or ["mon"]
        self.enable_earnings_calendar = cfg.enable_earnings_calendar
        self.peer_analysis_time = cfg.peer_analysis_time
        self.peer_analysis_days = cfg.peer_analysis_days or ["sun"]
        self.enable_peer_analysis = cfg.enable_peer_analysis
        self.correlation_audit_time = cfg.correlation_audit_time
        self.correlation_audit_days = cfg.correlation_audit_days or ["sun"]
        self.enable_correlation_audit = cfg.enable_correlation_audit
        self.tearsheet_time = cfg.tearsheet_time
        self.enable_reporting = cfg.enable_reporting
        self.rebalancing_time = cfg.rebalancing_time
        self.rebalancing_days = cfg.rebalancing_days or ["mon"]
        self.enable_rebalancing = cfg.enable_rebalancing
        self.signal_tracking_time = cfg.signal_tracking_time
        self.enable_signal_tracking = cfg.enable_signal_tracking
        self.game_plan_time = cfg.game_plan_time
        self.enable_game_plan = cfg.enable_game_plan
        self.pre_market_screening_time = cfg.pre_market_screening_time
        self.enable_pre_market_screening = cfg.enable_pre_market_screening
        self.monte_carlo_time = cfg.monte_carlo_time
        self.monte_carlo_days = cfg.monte_carlo_days or ["sun"]
        self.discovery_outcome_time = cfg.discovery_outcome_time
        self.discovery_outcome_days = cfg.discovery_outcome_days or ["mon", "tue", "wed", "thu", "fri"]
        self.enable_discovery_outcome = cfg.enable_discovery_outcome
        self._reddit_scraping_last_run: datetime | None = None
        logger.info(
            f"MarketScheduler initialized: {cfg.start_time}-{cfg.end_time} {cfg.timezone} "
            f"(pre-market={'enabled' if cfg.enable_pre_market else 'disabled'}, "
            f"after-hours={'enabled' if cfg.enable_after_hours else 'disabled'})"
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

    def get_current_phase_end(self) -> datetime | None:
        """Get when the current market phase ends.

        Returns:
            Datetime when current phase ends, None if market is closed
        """
        session = self.get_trading_session()
        if not session:
            return None

        now = datetime.now(self.timezone)

        if session == TradingSession.PRE_MARKET:
            # Pre-market ends when regular market opens
            return now.replace(
                hour=self.start_hour,
                minute=self.start_minute,
                second=0,
                microsecond=0,
            )
        if session == TradingSession.REGULAR:
            # Regular market ends at configured close time
            return now.replace(
                hour=self.end_hour,
                minute=self.end_minute,
                second=0,
                microsecond=0,
            )
        if session == TradingSession.AFTER_HOURS:
            # After-hours ends at 8:00 PM ET
            return now.replace(
                hour=AFTER_HOURS_END[0],
                minute=AFTER_HOURS_END[1],
                second=0,
                microsecond=0,
            )

        return None

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
            logger.opt(exception=True).warning(f"Malformed prefetch_time '{self.prefetch_time}': {e}")
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
            logger.opt(exception=True).warning(
                f"Malformed pre_market_refresh_time '{self.pre_market_refresh_time}': {e}"
            )
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
            logger.opt(exception=True).warning(f"Malformed tearsheet_time '{self.tearsheet_time}': {e}")
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
            logger.opt(exception=True).warning(
                f"Malformed signal_tracking_time '{self.signal_tracking_time}': {e}"
            )
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
            logger.opt(exception=True).warning(f"Malformed rebalancing_time '{self.rebalancing_time}': {e}")
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
            logger.opt(exception=True).warning(f"Malformed game_plan_time '{self.game_plan_time}': {e}")
            return False

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_pre_market_screening_time(self) -> bool:
        """Check if it's time for pre-market screening (07:00 ET, weekdays only).

        Returns:
            True if within 1 minute of configured time on weekday
        """
        if not self.enable_pre_market_screening:
            return False

        now = datetime.now(self.timezone)

        if now.weekday() >= 5:
            return False

        try:
            target_hour, target_minute = map(int, self.pre_market_screening_time.split(":"))
        except (ValueError, AttributeError) as e:
            logger.opt(exception=True).warning(
                f"Malformed pre_market_screening_time '{self.pre_market_screening_time}': {e}"
            )
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
        normalized_days = {day.lower() for day in self.monte_carlo_days}
        if current_day not in normalized_days:
            return False

        try:
            target_hour, target_minute = map(int, self.monte_carlo_time.split(":"))
        except (ValueError, AttributeError) as e:
            logger.opt(exception=True).warning(
                f"Invalid monte_carlo_time format: {self.monte_carlo_time}: {e}"
            )
            return False

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_discovery_outcome_time(self) -> bool:
        """Check if current time matches discovery outcome tracking schedule.

        Returns:
            True if current time is within 1 minute of configured time on configured day
        """
        if not self.enable_discovery_outcome:
            return False

        now = datetime.now(self.timezone)

        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]
        normalized_days = {day.lower() for day in self.discovery_outcome_days}
        if current_day not in normalized_days:
            return False

        try:
            target_hour, target_minute = map(int, self.discovery_outcome_time.split(":"))
        except (ValueError, AttributeError) as e:
            logger.opt(exception=True).warning(
                f"Invalid discovery_outcome_time format: {self.discovery_outcome_time}: {e}"
            )
            return False

        current_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        return abs(current_minutes - target_minutes) <= 1

    def is_reddit_scraping_time(self) -> bool:
        """Check if Reddit scraping should run (interval-based).

        Checks every cycle and uses internal state to track last run.
        Interval is hardcoded to 15 minutes (config reddit_scraper.interval_minutes
        not accessible here - task handles actual interval via dedup).

        Returns:
            True if should check (always True, task handles dedup)
        """
        # Always return True - let task handle interval-based dedup
        # This ensures task runs frequently enough to check its own schedule
        return True

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MarketScheduler({self.start_hour:02d}:{self.start_minute:02d}-"
            f"{self.end_hour:02d}:{self.end_minute:02d} {self.timezone})"
        )
