"""Tests for MarketService."""

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.daemon.market_service import MarketService
from src.daemon.scheduler import MarketScheduler
from src.strategies.session import TradingSession


@pytest.fixture
def scheduler() -> MarketScheduler:
    """Default MarketScheduler (09:30-16:00 ET)."""
    return MarketScheduler()


@pytest.fixture
def service(scheduler: MarketScheduler) -> MarketService:
    """MarketService backed by default scheduler."""
    return MarketService(scheduler)


@pytest.mark.unit
class TestMarketServiceInit:
    def test_repr_contains_class_name(self, service: MarketService) -> None:
        assert "MarketService" in repr(service)

    def test_repr_contains_open_time(self, service: MarketService) -> None:
        assert "09:30" in repr(service)


@pytest.mark.unit
class TestMarketServiceDelegation:
    def test_is_open_delegates_to_scheduler(self, scheduler: MarketScheduler) -> None:
        svc = MarketService(scheduler)
        scheduler.is_market_open = MagicMock(return_value=True)
        assert svc.is_open() is True
        scheduler.is_market_open.assert_called_once()

    def test_current_session_delegates_to_scheduler(self, scheduler: MarketScheduler) -> None:
        svc = MarketService(scheduler)
        scheduler.get_trading_session = MagicMock(return_value=TradingSession.REGULAR)
        assert svc.current_session() == TradingSession.REGULAR

    def test_is_regular_session_true_during_regular(self, scheduler: MarketScheduler) -> None:
        svc = MarketService(scheduler)
        scheduler.get_trading_session = MagicMock(return_value=TradingSession.REGULAR)
        assert svc.is_regular_session() is True

    def test_is_regular_session_false_outside_regular(self, scheduler: MarketScheduler) -> None:
        svc = MarketService(scheduler)
        scheduler.get_trading_session = MagicMock(return_value=None)
        assert svc.is_regular_session() is False

    def test_time_until_open_delegates_to_scheduler(self, scheduler: MarketScheduler) -> None:
        svc = MarketService(scheduler)
        scheduler.time_until_open = MagicMock(return_value=3600)
        assert svc.time_until_open() == 3600

    def test_time_until_close_delegates_to_scheduler(self, scheduler: MarketScheduler) -> None:
        svc = MarketService(scheduler)
        scheduler.time_until_close = MagicMock(return_value=7200)
        assert svc.time_until_close() == 7200


@pytest.mark.unit
class TestNextRegularOpen:
    """next_regular_open returns future 09:30 ET skipping weekends."""

    def test_before_open_returns_today(self, service: MarketService) -> None:
        """Before 09:30 on a weekday → same day 09:30."""
        tz = ZoneInfo("America/New_York")
        # Monday 08:00 → next open is 09:30 same day
        mock_now = datetime(2024, 1, 15, 8, 0, 0, tzinfo=tz)
        with patch("src.daemon.market_service.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            result = service.next_regular_open()
        assert result.hour == 9
        assert result.minute == 30
        assert result.date() == mock_now.date()

    def test_after_open_advances_to_next_day(self, service: MarketService) -> None:
        """After 09:30 on a weekday → next day 09:30."""
        tz = ZoneInfo("America/New_York")
        # Monday 12:00 → next open is Tuesday 09:30
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=tz)
        with patch("src.daemon.market_service.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            result = service.next_regular_open()
        assert result.weekday() == 1  # Tuesday
        assert result.hour == 9
        assert result.minute == 30

    def test_friday_evening_advances_to_monday(self, service: MarketService) -> None:
        """Friday evening → skips weekend → Monday 09:30."""
        tz = ZoneInfo("America/New_York")
        # Friday 2024-01-19 17:00
        mock_now = datetime(2024, 1, 19, 17, 0, 0, tzinfo=tz)
        with patch("src.daemon.market_service.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            result = service.next_regular_open()
        # Should land on Monday 2024-01-22
        assert result.weekday() == 0  # Monday
        assert result.day == 22
        assert result.hour == 9
        assert result.minute == 30

    def test_saturday_advances_to_monday(self, service: MarketService) -> None:
        """During weekend (Saturday) → Monday 09:30."""
        tz = ZoneInfo("America/New_York")
        mock_now = datetime(2024, 1, 20, 10, 0, 0, tzinfo=tz)
        with patch("src.daemon.market_service.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            result = service.next_regular_open()
        assert result.weekday() == 0  # Monday

    def test_sunday_advances_to_monday(self, service: MarketService) -> None:
        """During weekend (Sunday) → Monday 09:30."""
        tz = ZoneInfo("America/New_York")
        mock_now = datetime(2024, 1, 21, 10, 0, 0, tzinfo=tz)
        with patch("src.daemon.market_service.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            result = service.next_regular_open()
        assert result.weekday() == 0  # Monday

    def test_result_is_tz_aware(self, service: MarketService) -> None:
        """Result has timezone info attached."""
        tz = ZoneInfo("America/New_York")
        mock_now = datetime(2024, 1, 15, 8, 0, 0, tzinfo=tz)
        with patch("src.daemon.market_service.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            result = service.next_regular_open()
        assert result.tzinfo is not None
