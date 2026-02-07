"""Tests for daemon scheduler."""

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src.daemon.scheduler import MarketScheduler
from src.strategies.session import TradingSession


class TestMarketScheduler:
    def test_initialization(self):
        scheduler = MarketScheduler()

        assert scheduler.start_hour == 9
        assert scheduler.start_minute == 30
        assert scheduler.end_hour == 16
        assert scheduler.end_minute == 0
        assert scheduler.timezone == ZoneInfo("America/New_York")

    def test_custom_times(self):
        scheduler = MarketScheduler(
            start_time="10:00",
            end_time="15:30",
            timezone="America/Chicago",
        )

        assert scheduler.start_hour == 10
        assert scheduler.start_minute == 0
        assert scheduler.end_hour == 15
        assert scheduler.end_minute == 30
        assert scheduler.timezone == ZoneInfo("America/Chicago")

    def test_is_market_open_during_hours(self):
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")
        mock_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=tz)

        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_market_open() is True

    def test_is_market_closed_before_open(self):
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")
        mock_time = datetime(2024, 1, 15, 8, 0, 0, tzinfo=tz)

        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_market_open() is False

    def test_is_market_closed_after_close(self):
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")
        mock_time = datetime(2024, 1, 15, 17, 0, 0, tzinfo=tz)

        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_market_open() is False

    def test_is_market_closed_weekend(self):
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")
        mock_time = datetime(2024, 1, 13, 12, 0, 0, tzinfo=tz)

        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_market_open() is False

    def test_repr(self):
        scheduler = MarketScheduler(start_time="09:30", end_time="16:00")
        repr_str = repr(scheduler)

        assert "09:30" in repr_str
        assert "16:00" in repr_str

    def test_get_trading_session_pre_market(self):
        """Test pre-market session detection (4:00-9:30 AM)."""
        scheduler = MarketScheduler(enable_pre_market=True)
        tz = ZoneInfo("America/New_York")

        # 6:00 AM on Monday = PRE_MARKET
        mock_time = datetime(2024, 1, 15, 6, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            session = scheduler.get_trading_session()
            assert session == TradingSession.PRE_MARKET
            assert scheduler.is_market_open() is True

    def test_get_trading_session_regular(self):
        """Test regular session detection (9:30 AM-4:00 PM)."""
        scheduler = MarketScheduler(enable_pre_market=True)
        tz = ZoneInfo("America/New_York")

        # 12:00 PM on Monday = REGULAR
        mock_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            session = scheduler.get_trading_session()
            assert session == TradingSession.REGULAR

    def test_get_trading_session_disabled(self):
        """Test pre-market disabled (returns None before 9:30 AM)."""
        scheduler = MarketScheduler(enable_pre_market=False)
        tz = ZoneInfo("America/New_York")

        # 6:00 AM on Monday, pre-market disabled
        mock_time = datetime(2024, 1, 15, 6, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            session = scheduler.get_trading_session()
            assert session is None
            assert scheduler.is_market_open() is False

    def test_get_trading_session_weekend_pre_market(self):
        """Test pre-market disabled on weekends."""
        scheduler = MarketScheduler(enable_pre_market=True)
        tz = ZoneInfo("America/New_York")

        # 6:00 AM on Saturday (pre-market enabled but weekend)
        mock_time = datetime(2024, 1, 20, 6, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            session = scheduler.get_trading_session()
            assert session is None
            assert scheduler.is_market_open() is False

    def test_time_until_open_pre_market(self):
        """Test countdown to pre-market open (4:00 AM)."""
        scheduler = MarketScheduler(enable_pre_market=True)
        tz = ZoneInfo("America/New_York")

        # 2:00 AM on Monday (2 hours until pre-market)
        mock_time = datetime(2024, 1, 15, 2, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            wait_time = scheduler.time_until_open()
            assert wait_time == 2 * 3600  # 2 hours in seconds

    def test_pre_market_boundary_4am(self):
        """Test boundary: 4:00 AM = pre-market open."""
        scheduler = MarketScheduler(enable_pre_market=True)
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 4, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            assert scheduler.get_trading_session() == TradingSession.PRE_MARKET

    def test_pre_market_boundary_929am(self):
        """Test boundary: 9:29 AM = still pre-market."""
        scheduler = MarketScheduler(enable_pre_market=True)
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 9, 29, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            assert scheduler.get_trading_session() == TradingSession.PRE_MARKET

    def test_regular_market_boundary_930am(self):
        """Test boundary: 9:30 AM = regular market open."""
        scheduler = MarketScheduler(enable_pre_market=True)
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            assert scheduler.get_trading_session() == TradingSession.REGULAR

    def test_is_journal_window_during_window(self):
        """Test 16:15 ET is in journal window (default 15min offset)."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 16, 20, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            assert scheduler.is_journal_window() is True

    def test_is_journal_window_before_window(self):
        """Test 16:00 ET is before journal window."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 16, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            assert scheduler.is_journal_window() is False

    def test_is_journal_window_after_window(self):
        """Test 17:00 ET is after journal window."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 17, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            assert scheduler.is_journal_window() is False

    def test_is_journal_window_weekend(self):
        """Test journal window is inactive on weekends."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        # Saturday at 16:20
        mock_time = datetime(2024, 1, 20, 16, 20, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = datetime

            assert scheduler.is_journal_window() is False

    def test_is_after_hours_screening_time_enabled(self):
        """Test screening time matches configured time."""
        scheduler = MarketScheduler(
            enable_after_hours=True,
            after_hours_screen_time="16:30",
            after_hours_screen_days=["mon", "tue", "wed"],
        )
        tz = ZoneInfo("America/New_York")

        # Monday at 16:30 = match
        mock_time = datetime(2024, 1, 15, 16, 30, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_after_hours_screening_time() is True

    def test_is_after_hours_screening_time_tolerance(self):
        """Test ±1 minute tolerance."""
        scheduler = MarketScheduler(
            enable_after_hours=True,
            after_hours_screen_time="16:30",
        )
        tz = ZoneInfo("America/New_York")

        # 16:31 = within tolerance
        mock_time = datetime(2024, 1, 15, 16, 31, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_after_hours_screening_time() is True

        # 16:32 = outside tolerance
        mock_time = datetime(2024, 1, 15, 16, 32, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_after_hours_screening_time() is False

    def test_is_after_hours_screening_time_day_filter(self):
        """Test day filtering."""
        scheduler = MarketScheduler(
            enable_after_hours=True,
            after_hours_screen_time="16:30",
            after_hours_screen_days=["mon", "wed"],
        )
        tz = ZoneInfo("America/New_York")

        # Tuesday at 16:30 = not in allowed days
        mock_time = datetime(2024, 1, 16, 16, 30, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_after_hours_screening_time() is False

    def test_is_after_hours_screening_time_disabled(self):
        """Test screening disabled when enable_after_hours=False."""
        scheduler = MarketScheduler(
            enable_after_hours=False,
            after_hours_screen_time="16:30",
        )
        tz = ZoneInfo("America/New_York")

        # Monday at 16:30 but disabled
        mock_time = datetime(2024, 1, 15, 16, 30, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_after_hours_screening_time() is False


class TestHealthCheckTime:
    def test_match_default_time(self):
        """Test 17:00 matches default health check time."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 17, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_health_check_time() is True

    def test_tolerance_plus_one(self):
        """Test +1 minute tolerance."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 17, 1, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_health_check_time() is True

    def test_tolerance_minus_one(self):
        """Test -1 minute tolerance."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 16, 59, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_health_check_time() is True

    def test_miss_outside_tolerance(self):
        """Test 17:02 is outside tolerance."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 17, 2, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_health_check_time() is False

    def test_weekend_returns_false(self):
        """Test health check skipped on weekends."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        # Saturday at 17:00
        mock_time = datetime(2024, 1, 20, 17, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_health_check_time() is False

    def test_custom_time(self):
        """Test custom health check time."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 18, 30, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_health_check_time("18:30") is True
            assert scheduler.is_health_check_time("17:00") is False

    def test_malformed_time(self):
        """Test malformed health_run_time strings return False."""
        scheduler = MarketScheduler()
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 17, 0, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_health_check_time("bad") is False
            assert scheduler.is_health_check_time("17") is False
            assert scheduler.is_health_check_time("17:xx") is False
            assert scheduler.is_health_check_time("") is False
