"""Tests for fundamental worker."""

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.fundamental import FundamentalAnalysis
from src.data.earnings import EarningsCalendar, EarningsEvent
from src.tools.models import ToolDefinition
from src.workers.fundamental import FundamentalWorker


@pytest.fixture
def mock_fundamental_overview():
    """Alpha Vantage overview data."""
    return {
        "PERatio": "25.5",
        "EPS": "3.45",
        "QuarterlyRevenueGrowthYOY": "0.15",
        "QuarterlyEarningsGrowthYOY": "0.12",
        "DebtToEquity": "1.8",
        "CurrentRatio": "1.2",
    }


@pytest.fixture
def mock_earnings_calendar_upcoming():
    """Earnings 3 days out."""
    return EarningsCalendar(
        events=[
            EarningsEvent(
                symbol="AAPL",
                earnings_date=date.today() + timedelta(days=3),
                estimate_eps=3.50,
            )
        ],
        fetched_at=datetime.now(),
    )


@pytest.fixture
def mock_earnings_calendar_past():
    """Earnings 2 days ago."""
    return EarningsCalendar(
        events=[
            EarningsEvent(
                symbol="AAPL",
                earnings_date=date.today() - timedelta(days=2),
                estimate_eps=3.50,
            )
        ],
        fetched_at=datetime.now(),
    )


@pytest.fixture
def mock_earnings_calendar_far():
    """Earnings 7 days out (outside 5-day window)."""
    return EarningsCalendar(
        events=[
            EarningsEvent(
                symbol="AAPL",
                earnings_date=date.today() + timedelta(days=7),
                estimate_eps=3.50,
            )
        ],
        fetched_at=datetime.now(),
    )


@pytest.fixture
def mock_earnings_calendar_empty():
    """No earnings events."""
    return EarningsCalendar(events=[], fetched_at=datetime.now())


@pytest.fixture
def mock_fundamental_worker(test_container):
    """Create FundamentalWorker with mocked dependencies."""
    fundamental_fetcher = Mock()
    earnings_fetcher = Mock()
    llm_client = test_container.llm_client()
    return FundamentalWorker(llm_client, fundamental_fetcher, earnings_fetcher)


def test_fundamental_worker_init(mock_fundamental_worker):
    """Test worker initialization."""
    assert mock_fundamental_worker.llm is not None
    assert mock_fundamental_worker.fundamental_fetcher is not None
    assert mock_fundamental_worker.earnings_fetcher is not None

    tool_def = mock_fundamental_worker.get_tool_definition()
    assert tool_def is not None


def test_fundamental_worker_tool_definition(mock_fundamental_worker):
    """Test tool definition structure."""
    tool_def = mock_fundamental_worker.get_tool_definition()

    assert isinstance(tool_def, ToolDefinition)
    assert tool_def.type == "function"
    assert tool_def.function.name == "analyze_fundamental"
    assert "symbol" in tool_def.function.parameters.properties
    assert "current_price" in tool_def.function.parameters.properties
    assert "symbol" in tool_def.function.parameters.required


async def test_fundamental_worker_analyze_full_data(
    mock_fundamental_worker,
    mock_fundamental_overview,
    mock_earnings_calendar_upcoming,
):
    """Test analysis with full data."""
    # Setup mocks
    mock_fundamental_worker.fundamental_fetcher.fetch_overview = Mock(return_value=mock_fundamental_overview)
    mock_fundamental_worker.earnings_fetcher.fetch_earnings_dates = Mock(
        return_value=mock_earnings_calendar_upcoming
    )

    # Mock LLM response
    mock_llm_response = Mock(
        interpretation="Strong fundamentals with upcoming earnings catalyst.",
        confidence_keywords=["strong", "high confidence"],
    )
    mock_fundamental_worker.llm.astructured = AsyncMock(return_value=mock_llm_response)

    # Analyze
    result = await mock_fundamental_worker.analyze("AAPL", current_price=150.0)

    # Verify result structure
    assert isinstance(result, FundamentalAnalysis)
    assert result.valuation == "FAIRLY_VALUED"  # P/E 25.5 is in 15-30 range
    assert result.earnings_flags is not None
    assert result.earnings_flags.upcoming_earnings is True
    assert result.earnings_flags.days_until_earnings == 3
    assert result.pe_ratio == 25.5
    assert result.eps == 3.45
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation


async def test_fundamental_worker_earnings_flagging_upcoming(
    mock_fundamental_worker,
    mock_fundamental_overview,
    mock_earnings_calendar_upcoming,
):
    """Test earnings flagging: 3 days out (should flag)."""
    mock_fundamental_worker.fundamental_fetcher.fetch_overview = Mock(return_value=mock_fundamental_overview)
    mock_fundamental_worker.earnings_fetcher.fetch_earnings_dates = Mock(
        return_value=mock_earnings_calendar_upcoming
    )

    mock_llm_response = Mock(
        interpretation="Test interpretation.",
        confidence_keywords=["clear"],
    )
    mock_fundamental_worker.llm.astructured = AsyncMock(return_value=mock_llm_response)

    result = await mock_fundamental_worker.analyze("AAPL")

    assert result.earnings_flags.upcoming_earnings is True
    assert result.earnings_flags.days_until_earnings == 3
    assert result.earnings_flags.earnings_date == date.today() + timedelta(days=3)


async def test_fundamental_worker_earnings_flagging_past(
    mock_fundamental_worker,
    mock_fundamental_overview,
    mock_earnings_calendar_past,
):
    """Test earnings flagging: 2 days ago (should flag, within ±5 days)."""
    mock_fundamental_worker.fundamental_fetcher.fetch_overview = Mock(return_value=mock_fundamental_overview)
    mock_fundamental_worker.earnings_fetcher.fetch_earnings_dates = Mock(
        return_value=mock_earnings_calendar_past
    )

    mock_llm_response = Mock(
        interpretation="Test interpretation.",
        confidence_keywords=["clear"],
    )
    mock_fundamental_worker.llm.astructured = AsyncMock(return_value=mock_llm_response)

    result = await mock_fundamental_worker.analyze("AAPL")

    assert result.earnings_flags.upcoming_earnings is True
    assert result.earnings_flags.days_until_earnings == -2
    assert result.earnings_flags.earnings_date == date.today() - timedelta(days=2)


async def test_fundamental_worker_earnings_flagging_far(
    mock_fundamental_worker,
    mock_fundamental_overview,
    mock_earnings_calendar_far,
):
    """Test earnings flagging: 7 days out (should NOT flag, outside 5-day window)."""
    mock_fundamental_worker.fundamental_fetcher.fetch_overview = Mock(return_value=mock_fundamental_overview)
    mock_fundamental_worker.earnings_fetcher.fetch_earnings_dates = Mock(
        return_value=mock_earnings_calendar_far
    )

    mock_llm_response = Mock(
        interpretation="Test interpretation.",
        confidence_keywords=["clear"],
    )
    mock_fundamental_worker.llm.astructured = AsyncMock(return_value=mock_llm_response)

    result = await mock_fundamental_worker.analyze("AAPL")

    assert result.earnings_flags.upcoming_earnings is False
    assert result.earnings_flags.days_until_earnings is None
    assert result.earnings_flags.earnings_date is None


async def test_fundamental_worker_earnings_flagging_no_events(
    mock_fundamental_worker,
    mock_fundamental_overview,
    mock_earnings_calendar_empty,
):
    """Test earnings flagging: no events (should NOT flag)."""
    mock_fundamental_worker.fundamental_fetcher.fetch_overview = Mock(return_value=mock_fundamental_overview)
    mock_fundamental_worker.earnings_fetcher.fetch_earnings_dates = Mock(
        return_value=mock_earnings_calendar_empty
    )

    mock_llm_response = Mock(
        interpretation="Test interpretation.",
        confidence_keywords=["clear"],
    )
    mock_fundamental_worker.llm.astructured = AsyncMock(return_value=mock_llm_response)

    result = await mock_fundamental_worker.analyze("AAPL")

    assert result.earnings_flags.upcoming_earnings is False


async def test_fundamental_worker_missing_pe_ratio(mock_fundamental_worker, mock_earnings_calendar_empty):
    """Test handling of missing P/E ratio (should default to FAIRLY_VALUED)."""
    overview_no_pe = {
        "PERatio": None,
        "EPS": "3.45",
        "QuarterlyRevenueGrowthYOY": "0.15",
    }

    mock_fundamental_worker.fundamental_fetcher.fetch_overview = Mock(return_value=overview_no_pe)
    mock_fundamental_worker.earnings_fetcher.fetch_earnings_dates = Mock(
        return_value=mock_earnings_calendar_empty
    )

    mock_llm_response = Mock(
        interpretation="Limited data available.",
        confidence_keywords=["limited data", "uncertain"],
    )
    mock_fundamental_worker.llm.astructured = AsyncMock(return_value=mock_llm_response)

    result = await mock_fundamental_worker.analyze("AAPL")

    assert result.valuation == "FAIRLY_VALUED"
    assert result.pe_ratio is None


async def test_fundamental_worker_confidence_high_completeness(
    mock_fundamental_worker,
    mock_fundamental_overview,
    mock_earnings_calendar_empty,
):
    """Test confidence calculation with high data completeness and positive keywords."""
    mock_fundamental_worker.fundamental_fetcher.fetch_overview = Mock(return_value=mock_fundamental_overview)
    mock_fundamental_worker.earnings_fetcher.fetch_earnings_dates = Mock(
        return_value=mock_earnings_calendar_empty
    )

    mock_llm_response = Mock(
        interpretation="Strong analysis.",
        confidence_keywords=["strong", "high confidence", "clear"],
    )
    mock_fundamental_worker.llm.astructured = AsyncMock(return_value=mock_llm_response)

    result = await mock_fundamental_worker.analyze("AAPL")

    # High completeness (6/6 fields) + positive keywords should yield high confidence
    assert result.confidence >= 0.8


async def test_fundamental_worker_confidence_low_completeness(
    mock_fundamental_worker, mock_earnings_calendar_empty
):
    """Test confidence calculation with low data completeness and negative keywords."""
    sparse_overview = {
        "PERatio": "25.5",
        "EPS": None,
        "QuarterlyRevenueGrowthYOY": None,
        "QuarterlyEarningsGrowthYOY": None,
        "DebtToEquity": None,
        "CurrentRatio": None,
    }

    mock_fundamental_worker.fundamental_fetcher.fetch_overview = Mock(return_value=sparse_overview)
    mock_fundamental_worker.earnings_fetcher.fetch_earnings_dates = Mock(
        return_value=mock_earnings_calendar_empty
    )

    mock_llm_response = Mock(
        interpretation="Limited data, uncertain analysis.",
        confidence_keywords=["limited data", "uncertain"],
    )
    mock_fundamental_worker.llm.astructured = AsyncMock(return_value=mock_llm_response)

    result = await mock_fundamental_worker.analyze("AAPL")

    # Low completeness (1/6 fields) + negative keywords should yield low confidence
    assert result.confidence < 0.5


def test_fundamental_worker_calculate_earnings_flags_none_calendar(mock_fundamental_worker):
    """Test _calculate_earnings_flags with None calendar (fetch failure)."""
    flags = mock_fundamental_worker._calculate_earnings_flags(None, "AAPL")

    assert flags.upcoming_earnings is False
    assert flags.days_until_earnings is None
    assert flags.earnings_date is None


def test_fundamental_worker_parse_float(mock_fundamental_worker):
    """Test _parse_float utility."""
    assert mock_fundamental_worker._parse_float("25.5") == 25.5
    assert mock_fundamental_worker._parse_float(25.5) == 25.5
    assert mock_fundamental_worker._parse_float(None) is None
    assert mock_fundamental_worker._parse_float("-") is None
    assert mock_fundamental_worker._parse_float("None") is None
    assert mock_fundamental_worker._parse_float("invalid") is None


def test_fundamental_worker_repr(mock_fundamental_worker):
    """Test string representation."""
    repr_str = repr(mock_fundamental_worker)

    assert "FundamentalWorker" in repr_str
