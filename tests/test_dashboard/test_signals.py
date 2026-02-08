"""Tests for signals tab filtering logic."""

from datetime import UTC, datetime

import pytest

from src.dashboard.tabs.signals import _apply_filters


@pytest.fixture
def sample_analyses() -> list[dict]:
    """Sample analyses for testing."""
    return [
        {
            "symbol": "AAPL",
            "timestamp": datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC).isoformat(),
            "signal": "BUY",
            "confidence": 0.85,
        },
        {
            "symbol": "TSLA",
            "timestamp": datetime(2025, 1, 1, 11, 0, 0, tzinfo=UTC).isoformat(),
            "signal": "SELL",
            "confidence": 0.75,
        },
        {
            "symbol": "AAPL",
            "timestamp": datetime(2025, 1, 2, 9, 0, 0, tzinfo=UTC).isoformat(),
            "signal": "HOLD",
            "confidence": 0.60,
        },
        {
            "symbol": "MSFT",
            "timestamp": datetime(2025, 1, 3, 14, 0, 0, tzinfo=UTC).isoformat(),
            "signal": "BUY",
            "confidence": 0.90,
        },
        {
            "symbol": "GOOGL",
            "timestamp": datetime(2025, 1, 3, 15, 0, 0, tzinfo=UTC).isoformat(),
            "signal": "HOLD",
            "confidence": 0.55,
        },
    ]


def test_apply_filters_symbol_single(sample_analyses: list[dict]) -> None:
    """Test filtering by single symbol."""
    filtered = _apply_filters(sample_analyses, ["AAPL"], [], "", "")
    assert len(filtered) == 2
    assert all(a["symbol"] == "AAPL" for a in filtered)


def test_apply_filters_symbol_multiple(sample_analyses: list[dict]) -> None:
    """Test filtering by multiple symbols."""
    filtered = _apply_filters(sample_analyses, ["AAPL", "TSLA"], [], "", "")
    assert len(filtered) == 3
    symbols = {a["symbol"] for a in filtered}
    assert symbols == {"AAPL", "TSLA"}


def test_apply_filters_symbol_none_selected(sample_analyses: list[dict]) -> None:
    """Test with no symbol filter (all symbols pass)."""
    filtered = _apply_filters(sample_analyses, [], [], "", "")
    assert len(filtered) == 5


def test_apply_filters_signal_type(sample_analyses: list[dict]) -> None:
    """Test filtering by signal type."""
    # BUY signals only
    filtered = _apply_filters(sample_analyses, [], ["BUY"], "", "")
    assert len(filtered) == 2
    assert all(a["signal"] == "BUY" for a in filtered)

    # HOLD signals only
    filtered = _apply_filters(sample_analyses, [], ["HOLD"], "", "")
    assert len(filtered) == 2
    assert all(a["signal"] == "HOLD" for a in filtered)

    # Multiple signal types
    filtered = _apply_filters(sample_analyses, [], ["BUY", "SELL"], "", "")
    assert len(filtered) == 3
    assert all(a["signal"] in ["BUY", "SELL"] for a in filtered)


def test_apply_filters_date_range(sample_analyses: list[dict]) -> None:
    """Test filtering by date range with UTC timezone and inclusive end-date."""
    # Single day (inclusive end-date should capture all events on that day)
    filtered = _apply_filters(sample_analyses, [], [], "2025-01-01", "2025-01-01")
    assert len(filtered) == 2
    assert all(a["timestamp"].startswith("2025-01-01") for a in filtered)

    # Multi-day range
    filtered = _apply_filters(sample_analyses, [], [], "2025-01-01", "2025-01-02")
    assert len(filtered) == 3

    # Date range excluding some records
    filtered = _apply_filters(sample_analyses, [], [], "2025-01-03", "2025-01-03")
    assert len(filtered) == 2
    assert all(a["timestamp"].startswith("2025-01-03") for a in filtered)

    # Future date (no matches)
    filtered = _apply_filters(sample_analyses, [], [], "2025-12-01", "2025-12-31")
    assert len(filtered) == 0


def test_apply_filters_combined(sample_analyses: list[dict]) -> None:
    """Test filtering with multiple criteria."""
    # AAPL BUY signals on 2025-01-01
    filtered = _apply_filters(sample_analyses, ["AAPL"], ["BUY"], "2025-01-01", "2025-01-01")
    assert len(filtered) == 1
    assert filtered[0]["symbol"] == "AAPL"
    assert filtered[0]["signal"] == "BUY"

    # AAPL or MSFT, BUY signals, on Jan 1-3
    filtered = _apply_filters(sample_analyses, ["AAPL", "MSFT"], ["BUY"], "2025-01-01", "2025-01-03")
    assert len(filtered) == 2
    symbols = {a["symbol"] for a in filtered}
    assert symbols == {"AAPL", "MSFT"}
    assert all(a["signal"] == "BUY" for a in filtered)

    # Multiple symbols, multiple signals, date range
    filtered = _apply_filters(
        sample_analyses, ["AAPL", "TSLA", "GOOGL"], ["HOLD", "SELL"], "2025-01-01", "2025-01-03"
    )
    assert len(filtered) == 3
    assert all(a["signal"] in ["HOLD", "SELL"] for a in filtered)


def test_apply_filters_empty_filters(sample_analyses: list[dict]) -> None:
    """Test with no filters (all data passes)."""
    filtered = _apply_filters(sample_analyses, [], [], "", "")
    assert len(filtered) == 5


def test_apply_filters_no_matches(sample_analyses: list[dict]) -> None:
    """Test filters that produce no matches."""
    # Non-existent symbol
    filtered = _apply_filters(sample_analyses, ["NFLX"], [], "", "")
    assert len(filtered) == 0

    # AAPL with SELL signal (AAPL only has BUY/HOLD)
    filtered = _apply_filters(sample_analyses, ["AAPL"], ["SELL"], "", "")
    assert len(filtered) == 0

    # Date range with no data
    filtered = _apply_filters(sample_analyses, [], [], "2024-12-01", "2024-12-31")
    assert len(filtered) == 0


def test_apply_filters_timezone_handling(sample_analyses: list[dict]) -> None:
    """Test UTC timezone handling in date filtering."""
    # All sample timestamps are UTC, should work correctly
    filtered = _apply_filters(sample_analyses, [], [], "2025-01-01", "2025-01-03")
    assert len(filtered) == 5

    # Verify end-date is inclusive (23:59:59)
    filtered = _apply_filters(sample_analyses, [], [], "2025-01-03", "2025-01-03")
    assert len(filtered) == 2
    # Both events on 2025-01-03 should be included (14:00 and 15:00)
