"""Tests for v1 task scheduling helpers."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.v1.tasks.models import WEEKDAYS, DayOfWeek, DedupStrategy, TaskSchedule
from src.v1.tasks.scheduling import is_due

ET = ZoneInfo("America/New_York")


class TestIsDue:
    """Tests for is_due()."""

    @pytest.mark.unit
    def test_after_time_no_previous_run(self) -> None:
        """Due when past scheduled time and never ran."""
        schedule = TaskSchedule(time="04:00", days=WEEKDAYS)
        now = datetime(2026, 2, 18, 4, 30, tzinfo=ET)
        assert is_due(schedule, None, now) is True

    @pytest.mark.unit
    def test_exactly_at_time(self) -> None:
        """Due at exact scheduled time."""
        schedule = TaskSchedule(time="09:30", days=WEEKDAYS)
        now = datetime(2026, 2, 18, 9, 30, 0, tzinfo=ET)
        assert is_due(schedule, None, now) is True

    @pytest.mark.unit
    def test_before_scheduled_time(self) -> None:
        """Not due before scheduled time."""
        schedule = TaskSchedule(time="04:00", days=WEEKDAYS)
        now = datetime(2026, 2, 18, 3, 59, tzinfo=ET)
        assert is_due(schedule, None, now) is False

    @pytest.mark.unit
    def test_already_ran_today(self) -> None:
        """Not due if already ran today (DAILY dedup)."""
        schedule = TaskSchedule(time="04:00", days=WEEKDAYS, dedup=DedupStrategy.DAILY)
        now = datetime(2026, 2, 18, 9, 0, tzinfo=ET)
        last = datetime(2026, 2, 18, 4, 5, tzinfo=ET)
        assert is_due(schedule, last, now) is False

    @pytest.mark.unit
    def test_ran_yesterday(self) -> None:
        """Due if last run was yesterday."""
        schedule = TaskSchedule(time="04:00", days=WEEKDAYS, dedup=DedupStrategy.DAILY)
        now = datetime(2026, 2, 18, 4, 30, tzinfo=ET)
        last = datetime(2026, 2, 17, 4, 5, tzinfo=ET)
        assert is_due(schedule, last, now) is True

    @pytest.mark.unit
    def test_wrong_day(self) -> None:
        """Not due on excluded days."""
        schedule = TaskSchedule(time="04:00", days=WEEKDAYS)
        # Saturday
        now = datetime(2026, 2, 21, 4, 30, tzinfo=ET)
        assert is_due(schedule, None, now) is False

    @pytest.mark.unit
    def test_weekend_schedule(self) -> None:
        """Due on weekends when configured."""
        schedule = TaskSchedule(time="10:00", days=[DayOfWeek.SAT, DayOfWeek.SUN])
        now = datetime(2026, 2, 21, 10, 30, tzinfo=ET)
        assert is_due(schedule, None, now) is True

    @pytest.mark.unit
    def test_disabled(self) -> None:
        """Not due when disabled."""
        schedule = TaskSchedule(time="04:00", days=WEEKDAYS, enabled=False)
        now = datetime(2026, 2, 18, 4, 30, tzinfo=ET)
        assert is_due(schedule, None, now) is False

    @pytest.mark.unit
    def test_interval_within(self) -> None:
        """Not due if INTERVAL dedup and within interval."""
        schedule = TaskSchedule(
            time="04:00", days=WEEKDAYS, dedup=DedupStrategy.INTERVAL, dedup_interval_minutes=30
        )
        now = datetime(2026, 2, 18, 4, 20, tzinfo=ET)
        last = datetime(2026, 2, 18, 4, 5, tzinfo=ET)
        assert is_due(schedule, last, now) is False

    @pytest.mark.unit
    def test_interval_expired(self) -> None:
        """Due if INTERVAL dedup and interval expired."""
        schedule = TaskSchedule(
            time="04:00", days=WEEKDAYS, dedup=DedupStrategy.INTERVAL, dedup_interval_minutes=30
        )
        now = datetime(2026, 2, 18, 4, 40, tzinfo=ET)
        last = datetime(2026, 2, 18, 4, 5, tzinfo=ET)
        assert is_due(schedule, last, now) is True

    @pytest.mark.unit
    def test_none_strategy(self) -> None:
        """Always due with NONE strategy (past time, correct day)."""
        schedule = TaskSchedule(time="04:00", days=WEEKDAYS, dedup=DedupStrategy.NONE)
        now = datetime(2026, 2, 18, 4, 1, tzinfo=ET)
        last = now - timedelta(seconds=10)
        assert is_due(schedule, last, now) is True

    @pytest.mark.unit
    def test_late_in_day_still_due(self) -> None:
        """Due even hours after scheduled time (no window)."""
        schedule = TaskSchedule(time="04:00", days=WEEKDAYS)
        now = datetime(2026, 2, 18, 15, 0, tzinfo=ET)
        assert is_due(schedule, None, now) is True
