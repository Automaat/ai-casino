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


class TestSectorRotationTime:
    def test_match_configured_time(self):
        """Test sector rotation time matches configured time."""
        scheduler = MarketScheduler(
            enable_sector_rotation=True,
            sector_rotation_time="16:15",
            sector_rotation_days=["mon", "tue", "wed", "thu", "fri"],
        )
        tz = ZoneInfo("America/New_York")

        # Monday at 16:15 = match
        mock_time = datetime(2024, 1, 15, 16, 15, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_sector_rotation_time() is True

    def test_tolerance(self):
        """Test ±1 minute tolerance."""
        scheduler = MarketScheduler(
            enable_sector_rotation=True,
            sector_rotation_time="16:15",
        )
        tz = ZoneInfo("America/New_York")

        # 16:16 = within tolerance
        mock_time = datetime(2024, 1, 15, 16, 16, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_sector_rotation_time() is True

        # 16:17 = outside tolerance
        mock_time = datetime(2024, 1, 15, 16, 17, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_sector_rotation_time() is False

    def test_day_filter(self):
        """Test day filtering."""
        scheduler = MarketScheduler(
            enable_sector_rotation=True,
            sector_rotation_time="16:15",
            sector_rotation_days=["mon", "wed"],
        )
        tz = ZoneInfo("America/New_York")

        # Tuesday at 16:15 = not in allowed days
        mock_time = datetime(2024, 1, 16, 16, 15, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_sector_rotation_time() is False

    def test_disabled(self):
        """Test sector rotation disabled."""
        scheduler = MarketScheduler(
            enable_sector_rotation=False,
            sector_rotation_time="16:15",
        )
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 16, 15, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_sector_rotation_time() is False


class TestEarningsFetchTime:
    def test_match_configured_time(self):
        """Test earnings fetch time matches configured time."""
        scheduler = MarketScheduler(
            enable_earnings_calendar=True,
            earnings_fetch_time="16:45",
            earnings_fetch_days=["mon", "tue", "wed", "thu", "fri"],
        )
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 16, 45, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_earnings_fetch_time() is True

    def test_tolerance(self):
        """Test ±1 minute tolerance."""
        scheduler = MarketScheduler(
            enable_earnings_calendar=True,
            earnings_fetch_time="16:45",
        )
        tz = ZoneInfo("America/New_York")

        # 16:46 = within tolerance
        mock_time = datetime(2024, 1, 15, 16, 46, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_earnings_fetch_time() is True

        # 16:47 = outside tolerance
        mock_time = datetime(2024, 1, 15, 16, 47, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_earnings_fetch_time() is False

    def test_wrong_day(self):
        """Test day filtering (default is mon only)."""
        scheduler = MarketScheduler(
            enable_earnings_calendar=True,
            earnings_fetch_time="16:45",
            earnings_fetch_days=["mon"],
        )
        tz = ZoneInfo("America/New_York")

        # Tuesday at 16:45 = not in allowed days
        mock_time = datetime(2024, 1, 16, 16, 45, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_earnings_fetch_time() is False

    def test_disabled(self):
        """Test earnings calendar disabled."""
        scheduler = MarketScheduler(
            enable_earnings_calendar=False,
            earnings_fetch_time="16:45",
        )
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 15, 16, 45, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_earnings_fetch_time() is False


class TestPeerAnalysisTime:
    def test_match_configured_time(self):
        """Test peer analysis time matches configured time on Sunday."""
        scheduler = MarketScheduler(
            enable_peer_analysis=True,
            peer_analysis_time="17:30",
            peer_analysis_days=["sun"],
        )
        tz = ZoneInfo("America/New_York")

        # Sunday at 17:30
        mock_time = datetime(2024, 1, 14, 17, 30, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_peer_analysis_time() is True

    def test_tolerance(self):
        """Test ±1 minute tolerance."""
        scheduler = MarketScheduler(
            enable_peer_analysis=True,
            peer_analysis_time="17:30",
            peer_analysis_days=["sun"],
        )
        tz = ZoneInfo("America/New_York")

        # 17:31 = within tolerance
        mock_time = datetime(2024, 1, 14, 17, 31, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_peer_analysis_time() is True

        # 17:32 = outside tolerance
        mock_time = datetime(2024, 1, 14, 17, 32, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_peer_analysis_time() is False

    def test_wrong_day(self):
        """Test day filtering (default is sun only)."""
        scheduler = MarketScheduler(
            enable_peer_analysis=True,
            peer_analysis_time="17:30",
            peer_analysis_days=["sun"],
        )
        tz = ZoneInfo("America/New_York")

        # Monday at 17:30 = not in allowed days
        mock_time = datetime(2024, 1, 15, 17, 30, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_peer_analysis_time() is False

    def test_disabled(self):
        """Test peer analysis disabled."""
        scheduler = MarketScheduler(
            enable_peer_analysis=False,
            peer_analysis_time="17:30",
        )
        tz = ZoneInfo("America/New_York")

        mock_time = datetime(2024, 1, 14, 17, 30, 0, tzinfo=tz)
        with patch("src.daemon.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time

            assert scheduler.is_peer_analysis_time() is False
