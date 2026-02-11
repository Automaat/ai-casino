"""Tests for daemon earnings calendar wrapper."""

from datetime import date
from unittest.mock import MagicMock, patch

from src.daemon.earnings import DaemonEarningsCalendar
from src.data.earnings import EarningsEvent


@patch("src.daemon.earnings.datetime")
class TestGetUpcoming:
    def test_within_window(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        events = [
            EarningsEvent(symbol="AAPL", earnings_date=date(2024, 7, 22)),
            EarningsEvent(symbol="TSLA", earnings_date=date(2024, 7, 30)),
        ]

        daemon = DaemonEarningsCalendar()
        upcoming = daemon.get_upcoming(events, days_ahead=3)

        assert len(upcoming) == 1
        assert upcoming[0].symbol == "AAPL"

    def test_outside_window(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        events = [EarningsEvent(symbol="AAPL", earnings_date=date(2024, 8, 1))]

        daemon = DaemonEarningsCalendar()
        upcoming = daemon.get_upcoming(events, days_ahead=3)

        assert len(upcoming) == 0

    def test_past_events_excluded(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        events = [EarningsEvent(symbol="AAPL", earnings_date=date(2024, 7, 18))]

        daemon = DaemonEarningsCalendar()
        upcoming = daemon.get_upcoming(events, days_ahead=3)

        assert len(upcoming) == 0

    def test_today_included(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        events = [EarningsEvent(symbol="AAPL", earnings_date=date(2024, 7, 20))]

        daemon = DaemonEarningsCalendar()
        upcoming = daemon.get_upcoming(events, days_ahead=3)

        assert len(upcoming) == 1


@patch("src.daemon.earnings.datetime")
class TestFormatContext:
    def test_format_single_event(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        upcoming = [EarningsEvent(symbol="AAPL", earnings_date=date(2024, 7, 22))]

        daemon = DaemonEarningsCalendar()
        context = daemon.format_context(upcoming)

        assert "AAPL" in context
        assert "2024-07-22" in context
        assert "2d away" in context

    def test_format_with_eps(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        upcoming = [EarningsEvent(symbol="MSFT", earnings_date=date(2024, 7, 21), estimate_eps=2.93)]

        daemon = DaemonEarningsCalendar()
        context = daemon.format_context(upcoming)

        assert "$2.93" in context

    def test_format_empty(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        daemon = DaemonEarningsCalendar()
        context = daemon.format_context([])

        assert context == ""


@patch("src.daemon.earnings.datetime")
class TestGetEarningsFlags:
    def test_symbol_with_upcoming_earnings(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        events = [EarningsEvent(symbol="AAPL", earnings_date=date(2024, 7, 22))]

        daemon = DaemonEarningsCalendar()
        flags = daemon.get_earnings_flags(events, "AAPL")

        assert flags.upcoming_earnings is True
        assert flags.days_until_earnings == 2
        assert flags.pre_earnings_zone == "T-3"

    def test_symbol_t1_zone(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        events = [EarningsEvent(symbol="AAPL", earnings_date=date(2024, 7, 21))]

        daemon = DaemonEarningsCalendar()
        flags = daemon.get_earnings_flags(events, "AAPL")

        assert flags.pre_earnings_zone == "T-1"

    def test_symbol_no_upcoming(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        events = [EarningsEvent(symbol="TSLA", earnings_date=date(2024, 8, 15))]

        daemon = DaemonEarningsCalendar()
        flags = daemon.get_earnings_flags(events, "AAPL")

        assert flags.upcoming_earnings is False
        assert flags.days_until_earnings is None
        assert flags.pre_earnings_zone is None

    def test_symbol_far_future(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        events = [EarningsEvent(symbol="AAPL", earnings_date=date(2024, 10, 25))]

        daemon = DaemonEarningsCalendar()
        flags = daemon.get_earnings_flags(events, "AAPL")

        assert flags.upcoming_earnings is True
        assert flags.days_until_earnings > 3
        assert flags.pre_earnings_zone is None

    def test_past_earnings_ignored(self, mock_dt):
        mock_dt.now.return_value = MagicMock(date=MagicMock(return_value=date(2024, 7, 20)))

        events = [EarningsEvent(symbol="AAPL", earnings_date=date(2024, 7, 18))]

        daemon = DaemonEarningsCalendar()
        flags = daemon.get_earnings_flags(events, "AAPL")

        assert flags.upcoming_earnings is False


class TestRepr:
    def test_repr(self):
        daemon = DaemonEarningsCalendar()
        assert "DaemonEarningsCalendar" in repr(daemon)
