"""Tests for events tab filtering logic."""

import pytest

from src.dashboard.tabs.events import _apply_event_filters, _categorize_event


@pytest.fixture
def sample_events() -> list[dict]:
    """Sample events for testing."""
    return [
        {
            "timestamp": "2025-01-01T10:00:00Z",
            "event_type": "analysis_complete",
            "source": "system",
            "data": {},
            "summary": "Analysis done",
        },
        {
            "timestamp": "2025-01-01T11:00:00+00:00",
            "event_type": "api_error",
            "source": "system",
            "data": {},
            "summary": "API failed",
        },
        {
            "timestamp": "2025-01-02T09:00:00Z",
            "event_type": "news",
            "source": "market",
            "data": {},
            "summary": "Market news",
        },
        {
            "timestamp": "2025-01-03T14:00:00Z",
            "event_type": "anomaly",
            "source": "market",
            "data": {},
            "summary": "Price spike",
        },
        {
            "timestamp": None,  # Missing timestamp
            "event_type": "cycle_start",
            "source": "system",
            "data": {},
            "summary": "Daemon started",
        },
    ]


def test_apply_event_filters_type_only(sample_events: list[dict]) -> None:
    """Test filtering by event type category."""
    # Filter for ANALYSIS category
    filtered = _apply_event_filters(sample_events, ["ANALYSIS"], "", "")
    assert len(filtered) == 1
    assert filtered[0]["event_type"] == "analysis_complete"

    # Filter for ERROR category
    filtered = _apply_event_filters(sample_events, ["ERROR"], "", "")
    assert len(filtered) == 1
    assert "error" in filtered[0]["event_type"].lower()

    # Filter for market events (NEWS)
    filtered = _apply_event_filters(sample_events, ["NEWS"], "", "")
    assert len(filtered) == 1
    assert filtered[0]["event_type"] == "news"

    # Multiple categories
    filtered = _apply_event_filters(sample_events, ["NEWS", "ANOMALY"], "", "")
    assert len(filtered) == 2
    assert {e["event_type"] for e in filtered} == {"news", "anomaly"}


def test_apply_event_filters_date_only(sample_events: list[dict]) -> None:
    """Test filtering by date range with inclusive end-date."""
    # Single day (inclusive end-date should capture events until 23:59:59)
    filtered = _apply_event_filters(sample_events, [], "2025-01-01", "2025-01-01")
    assert len(filtered) == 2
    assert all("2025-01-01" in e["timestamp"] for e in filtered)

    # Multi-day range
    filtered = _apply_event_filters(sample_events, [], "2025-01-01", "2025-01-02")
    assert len(filtered) == 3
    timestamps = [e["timestamp"] for e in filtered]
    assert any("2025-01-01" in ts for ts in timestamps)
    assert any("2025-01-02" in ts for ts in timestamps)

    # Future date (no matches)
    filtered = _apply_event_filters(sample_events, [], "2025-12-01", "2025-12-31")
    assert len(filtered) == 0


def test_apply_event_filters_combined(sample_events: list[dict]) -> None:
    """Test filtering with both type and date filters."""
    # ANALYSIS events on 2025-01-01
    filtered = _apply_event_filters(sample_events, ["ANALYSIS"], "2025-01-01", "2025-01-01")
    assert len(filtered) == 1
    assert filtered[0]["event_type"] == "analysis_complete"

    # Market events (NEWS + ANOMALY) from Jan 2-3
    filtered = _apply_event_filters(sample_events, ["NEWS", "ANOMALY"], "2025-01-02", "2025-01-03")
    assert len(filtered) == 2
    assert {e["event_type"] for e in filtered} == {"news", "anomaly"}

    # ERROR events on Jan 2 (no matches)
    filtered = _apply_event_filters(sample_events, ["ERROR"], "2025-01-02", "2025-01-02")
    assert len(filtered) == 0


def test_apply_event_filters_missing_timestamp(sample_events: list[dict]) -> None:
    """Test graceful handling of events with None timestamps."""
    # Date filter excludes events without timestamps
    filtered = _apply_event_filters(sample_events, [], "2025-01-01", "2025-01-03")
    assert len(filtered) == 4  # 4 events with timestamps

    # Event with None timestamp excluded from date filter
    assert not any(e["timestamp"] is None for e in filtered)

    # Type filter works independently (includes None timestamp event)
    filtered = _apply_event_filters(sample_events, ["SYSTEM"], "", "")
    system_events = [e for e in filtered if e["event_type"] == "cycle_start"]
    assert len(system_events) == 1


def test_apply_event_filters_z_suffix(sample_events: list[dict]) -> None:
    """Test ISO 8601 'Z' suffix handling."""
    # Events with 'Z' suffix should be correctly parsed
    filtered = _apply_event_filters(sample_events, [], "2025-01-01", "2025-01-01")
    assert len(filtered) == 2

    # Verify both Z and +00:00 formats are included
    timestamps = [e["timestamp"] for e in filtered]
    assert any(ts.endswith("Z") for ts in timestamps)
    assert any("+00:00" in ts for ts in timestamps)


def test_apply_event_filters_empty_filters(sample_events: list[dict]) -> None:
    """Test with no filters (all events pass)."""
    filtered = _apply_event_filters(sample_events, [], "", "")
    assert len(filtered) == 5  # All events


def test_categorize_event_market_events() -> None:
    """Test categorization of market events."""
    assert _categorize_event({"event_type": "news"}) == "NEWS"
    assert _categorize_event({"event_type": "social"}) == "SOCIAL"
    assert _categorize_event({"event_type": "anomaly"}) == "ANOMALY"
    assert _categorize_event({"event_type": "filing"}) == "FILING"


def test_categorize_event_system_events() -> None:
    """Test categorization of system events."""
    assert _categorize_event({"event_type": "analysis_complete"}) == "ANALYSIS"
    assert _categorize_event({"event_type": "trade_executed"}) == "ANALYSIS"
    assert _categorize_event({"event_type": "api_error"}) == "ERROR"
    assert _categorize_event({"event_type": "data_error"}) == "ERROR"
    assert _categorize_event({"event_type": "cycle_start"}) == "SYSTEM"
    assert _categorize_event({"event_type": "health_check"}) == "SYSTEM"
    assert _categorize_event({"event_type": "degradation"}) == "ERROR"
    assert _categorize_event({"event_type": "unknown_event"}) == "SYSTEM"


def test_categorize_event_case_insensitive() -> None:
    """Test case-insensitive event type handling."""
    assert _categorize_event({"event_type": "NEWS"}) == "NEWS"
    assert _categorize_event({"event_type": "News"}) == "NEWS"
    assert _categorize_event({"event_type": "API_ERROR"}) == "ERROR"
    assert _categorize_event({"event_type": "api_error"}) == "ERROR"


def test_categorize_event_missing_type() -> None:
    """Test categorization with missing event_type."""
    assert _categorize_event({}) == "SYSTEM"  # Default category
    assert _categorize_event({"event_type": ""}) == "SYSTEM"
