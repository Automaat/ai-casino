"""Tests for earnings calendar data fetcher."""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

from src.data.earnings import EarningsCalendar, EarningsCalendarFetcher, EarningsEvent


class TestEarningsEvent:
    def test_basic_event(self):
        event = EarningsEvent(symbol="AAPL", earnings_date=date(2024, 7, 25))

        assert event.symbol == "AAPL"
        assert event.earnings_date == date(2024, 7, 25)
        assert event.estimate_eps is None

    def test_event_with_eps(self):
        event = EarningsEvent(symbol="TSLA", earnings_date=date(2024, 7, 23), estimate_eps=0.62)

        assert event.estimate_eps == 0.62


class TestEarningsCalendar:
    def test_empty_calendar(self):
        cal = EarningsCalendar(events=[], fetched_at=datetime.now())

        assert cal.events == []
        assert cal.fetched_at is not None


class TestEarningsCalendarFetcher:
    @patch("src.data.earnings.yf.Ticker")
    def test_fetch_earnings_dates(self, mock_ticker_cls):
        future_date = date.today() + timedelta(days=10)
        mock_ticker = MagicMock()
        mock_ticker.calendar = {
            "Earnings Date": [datetime(future_date.year, future_date.month, future_date.day)]
        }
        mock_ticker_cls.return_value = mock_ticker

        fetcher = EarningsCalendarFetcher(delay_seconds=0)
        result = fetcher.fetch_earnings_dates(["AAPL"])

        assert isinstance(result, EarningsCalendar)
        assert len(result.events) == 1
        assert result.events[0].symbol == "AAPL"
        assert result.events[0].earnings_date == future_date

    @patch("src.data.earnings.yf.Ticker")
    def test_fetch_handles_missing_calendar(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.calendar = None
        mock_ticker_cls.return_value = mock_ticker

        fetcher = EarningsCalendarFetcher(delay_seconds=0)
        result = fetcher.fetch_earnings_dates(["UNKNOWN"])

        assert len(result.events) == 0

    @patch("src.data.earnings.yf.Ticker")
    def test_fetch_handles_exception(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("API error")

        fetcher = EarningsCalendarFetcher(delay_seconds=0)
        result = fetcher.fetch_earnings_dates(["BAD"])

        assert len(result.events) == 0

    @patch("src.data.earnings.yf.Ticker")
    def test_fetch_with_eps_estimate(self, mock_ticker_cls):
        future_date = date.today() + timedelta(days=5)
        mock_ticker = MagicMock()
        mock_ticker.calendar = {
            "Earnings Date": [datetime(future_date.year, future_date.month, future_date.day)],
            "Earnings Average": 1.25,
        }
        mock_ticker_cls.return_value = mock_ticker

        fetcher = EarningsCalendarFetcher(delay_seconds=0)
        result = fetcher.fetch_earnings_dates(["MSFT"])

        assert len(result.events) == 1
        assert result.events[0].estimate_eps == 1.25

    @patch("src.data.earnings.yf.Ticker")
    def test_fetch_multiple_symbols(self, mock_ticker_cls):
        future_date = date.today() + timedelta(days=7)

        def side_effect(symbol):
            mock = MagicMock()
            if symbol == "AAPL":
                mock.calendar = {
                    "Earnings Date": [datetime(future_date.year, future_date.month, future_date.day)]
                }
            else:
                mock.calendar = None
            return mock

        mock_ticker_cls.side_effect = side_effect

        fetcher = EarningsCalendarFetcher(delay_seconds=0)
        result = fetcher.fetch_earnings_dates(["AAPL", "UNKNOWN"])

        assert len(result.events) == 1
        assert result.events[0].symbol == "AAPL"

    def test_repr(self):
        fetcher = EarningsCalendarFetcher(delay_seconds=1.0)
        assert "1.0" in repr(fetcher)
