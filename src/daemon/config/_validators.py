"""Shared validation utilities for daemon config."""

import re
from typing import Literal

TIME_FORMAT_PATTERN = re.compile(r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$")


def validate_time_format(time_str: str, field_name: str) -> tuple[int, int]:
    """Validate HH:MM format, return (hour, minute).

    Args:
        time_str: Time string in HH:MM format
        field_name: Name of field being validated (for error messages)

    Returns:
        Tuple of (hour, minute) as integers

    Raises:
        ValueError: If time_str is not in valid HH:MM format
    """
    match = TIME_FORMAT_PATTERN.match(time_str)
    if not match:
        msg = f"{field_name} must be in HH:MM format (00:00-23:59), got {time_str}"
        raise ValueError(msg)
    return int(match.group(1)), int(match.group(2))


def validate_time_range(
    time_str: str,
    field_name: str,
    range_type: Literal["after_hours", "pre_market", "any"] = "after_hours",
) -> tuple[int, int]:
    """Validate time format and range.

    Args:
        time_str: Time string in HH:MM format
        field_name: Name of field being validated (for error messages)
        range_type: Expected time range - after_hours (16:00-20:00),
                   pre_market (04:00-09:30), or any (00:00-23:59)

    Returns:
        Tuple of (hour, minute) as integers

    Raises:
        ValueError: If time_str is invalid or outside expected range
    """
    hour, minute = validate_time_format(time_str, field_name)

    if range_type == "after_hours" and not (16 <= hour < 20 or (hour == 20 and minute == 0)):
        msg = f"{field_name} must be between 16:00-20:00, got {time_str}"
        raise ValueError(msg)
    if range_type == "pre_market" and not (4 <= hour < 9 or (hour == 9 and minute <= 30)):
        msg = f"{field_name} must be 04:00-09:30, got {time_str}"
        raise ValueError(msg)

    return hour, minute
