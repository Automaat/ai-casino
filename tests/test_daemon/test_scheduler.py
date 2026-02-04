"""Tests for daemon scheduler."""

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src.daemon.scheduler import MarketScheduler


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
