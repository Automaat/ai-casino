"""Schedule evaluation helpers."""

from datetime import datetime, timedelta

from src.v1.tasks.models import DayOfWeek, DedupStrategy, TaskSchedule

_DAY_MAP: dict[int, DayOfWeek] = {
    0: DayOfWeek.MON,
    1: DayOfWeek.TUE,
    2: DayOfWeek.WED,
    3: DayOfWeek.THU,
    4: DayOfWeek.FRI,
    5: DayOfWeek.SAT,
    6: DayOfWeek.SUN,
}


def is_due(schedule: TaskSchedule, last_run: datetime | None, now: datetime) -> bool:
    """Task is due if: enabled, correct day, past scheduled time, not already completed.

    Args:
        schedule: Task schedule
        last_run: Last execution timestamp (None = never ran)
        now: Current datetime (tz-aware)

    Returns:
        True if task should execute now
    """
    if not schedule.enabled:
        return False

    today_dow = _DAY_MAP[now.weekday()]
    if today_dow not in schedule.days:
        return False

    if schedule.dedup != DedupStrategy.INTERVAL:
        if schedule.time is None:
            return False
        hour, minute = map(int, schedule.time.split(":"))
        scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now < scheduled:
            return False

    return not _should_skip(schedule, last_run, now)


def _should_skip(schedule: TaskSchedule, last_run: datetime | None, now: datetime) -> bool:
    """Check dedup strategy against last_run timestamp."""
    if last_run is None:
        return False

    if schedule.dedup == DedupStrategy.NONE:
        return False

    if schedule.dedup == DedupStrategy.DAILY:
        return last_run.date() == now.date()

    if schedule.dedup == DedupStrategy.INTERVAL:
        interval = schedule.dedup_interval_minutes or 60
        return (now - last_run) < timedelta(minutes=interval)

    return False
