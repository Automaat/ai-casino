"""Tests for signal analytics API routes."""

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_summary_endpoint(async_client: AsyncClient):
    """Test signal flow summary endpoint."""
    start_date = (datetime.now(UTC) - timedelta(days=7)).date().isoformat()
    end_date = datetime.now(UTC).date().isoformat()

    response = await async_client.get(
        f"/api/signal-analytics/summary?start_date={start_date}&end_date={end_date}"
    )

    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "total_signals" in data
    assert "total_buy_signals" in data
    assert "total_sell_signals" in data
    assert "execution_rate" in data
    assert "executed_count" in data
    assert "not_executed_count" in data
    assert "overall_accuracy" in data
    assert "avg_confidence" in data
    assert "date_range" in data

    # Check types
    assert isinstance(data["total_signals"], int)
    assert isinstance(data["execution_rate"], float)
    assert isinstance(data["date_range"], list)
    assert len(data["date_range"]) == 2


@pytest.mark.asyncio
async def test_get_sankey_endpoint(async_client: AsyncClient):
    """Test Sankey flow data endpoint."""
    start_date = (datetime.now(UTC) - timedelta(days=7)).date().isoformat()
    end_date = datetime.now(UTC).date().isoformat()

    response = await async_client.get(
        f"/api/signal-analytics/sankey?start_date={start_date}&end_date={end_date}"
    )

    assert response.status_code == 200
    data = response.json()

    # Check structure
    assert "nodes" in data
    assert "links" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["links"], list)


@pytest.mark.asyncio
async def test_get_accuracy_by_type_endpoint(async_client: AsyncClient):
    """Test accuracy by type endpoint."""
    start_date = (datetime.now(UTC) - timedelta(days=7)).date().isoformat()
    end_date = datetime.now(UTC).date().isoformat()

    response = await async_client.get(
        f"/api/signal-analytics/accuracy-by-type?start_date={start_date}&end_date={end_date}&horizon=5d"
    )

    assert response.status_code == 200
    data = response.json()

    assert "data" in data
    assert "count" in data
    assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_get_calibration_endpoint(async_client: AsyncClient):
    """Test calibration curve endpoint."""
    start_date = (datetime.now(UTC) - timedelta(days=7)).date().isoformat()
    end_date = datetime.now(UTC).date().isoformat()

    response = await async_client.get(
        f"/api/signal-analytics/calibration?start_date={start_date}&end_date={end_date}&horizon=5d"
    )

    assert response.status_code == 200
    data = response.json()

    assert "buckets" in data
    assert isinstance(data["buckets"], list)


@pytest.mark.asyncio
async def test_get_timing_endpoint(async_client: AsyncClient):
    """Test timing analysis endpoint."""
    start_date = (datetime.now(UTC) - timedelta(days=7)).date().isoformat()
    end_date = datetime.now(UTC).date().isoformat()

    response = await async_client.get(
        f"/api/signal-analytics/timing?start_date={start_date}&end_date={end_date}"
    )

    assert response.status_code == 200
    data = response.json()

    assert "avg_execution_delay_hours" in data
    assert "by_confidence_bucket" in data
    assert isinstance(data["avg_execution_delay_hours"], float)
    assert isinstance(data["by_confidence_bucket"], dict)


@pytest.mark.asyncio
async def test_get_execution_rate_endpoint(async_client: AsyncClient):
    """Test execution rate endpoint."""
    start_date = (datetime.now(UTC) - timedelta(days=7)).date().isoformat()
    end_date = datetime.now(UTC).date().isoformat()

    response = await async_client.get(
        f"/api/signal-analytics/execution-rate?start_date={start_date}&end_date={end_date}"
    )

    assert response.status_code == 200
    data = response.json()

    assert "data" in data
    assert "count" in data
    assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_invalid_date_format_returns_400(async_client: AsyncClient):
    """Test that invalid date format returns 400."""
    response = await async_client.get("/api/signal-analytics/summary?start_date=invalid&end_date=2024-01-01")

    assert response.status_code == 400
    assert "Invalid date format" in response.json()["detail"]


@pytest.mark.asyncio
async def test_invalid_horizon_returns_400(async_client: AsyncClient):
    """Test that invalid horizon returns 400."""
    start_date = (datetime.now(UTC) - timedelta(days=7)).date().isoformat()
    end_date = datetime.now(UTC).date().isoformat()

    response = await async_client.get(
        f"/api/signal-analytics/accuracy-by-type?start_date={start_date}&end_date={end_date}&horizon=invalid"
    )

    assert response.status_code == 400
