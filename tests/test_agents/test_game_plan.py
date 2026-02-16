"""Tests for game plan agent."""

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.game_plan import GamePlan, GamePlanLLMResponse, KeyLevel
from src.models.providers.base import StructuredOutputError


@pytest.fixture
def sample_futures_data():
    """Sample futures data."""
    return {"ES=F": 0.5, "NQ=F": -0.3}


@pytest.fixture
def sample_game_plan():
    """Sample game plan."""
    return GamePlan(
        date=date.today(),
        priority_symbols=["AAPL", "TSLA", "NVDA"],
        risk_stance="NEUTRAL",
        sector_focus=["Technology"],
        key_levels={"AAPL": 175.0, "TSLA": 200.0},
        overnight_summary="Futures flat, Asia mixed",
        reasoning="Tech strength overnight, momentum intact",
        confidence=0.75,
        generated_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_market_fetcher():
    """Mock market data fetcher."""
    fetcher = MagicMock()
    fetcher.fetch_overnight_futures = MagicMock(return_value={"ES=F": 0.5, "NQ=F": -0.3})
    return fetcher


@patch("yfinance.Ticker")
async def test_game_plan_agent_generate(mock_yf_ticker, test_container, mock_market_fetcher):
    """Test game plan generation."""
    import pandas as pd

    agent = test_container.game_plan_agent()
    agent.market_fetcher = mock_market_fetcher
    agent.llm.astructured = AsyncMock(
        return_value=GamePlanLLMResponse(
            priority_symbols=["AAPL", "TSLA"],
            risk_stance="NEUTRAL",
            sector_focus=["Technology"],
            key_levels=[KeyLevel(symbol="AAPL", price=175.0)],
            reasoning="Tech strength overnight",
            confidence=0.8,
        )
    )

    mock_ticker = MagicMock()
    mock_history_df = pd.DataFrame({"Close": [100.0, 102.0]})
    mock_ticker.history.return_value = mock_history_df
    mock_yf_ticker.return_value = mock_ticker

    plan = await agent.generate(["AAPL", "TSLA", "GOOGL"])

    assert isinstance(plan, GamePlan)
    assert plan.priority_symbols == ["AAPL", "TSLA"]
    assert plan.risk_stance == "NEUTRAL"
    assert 0.0 <= plan.confidence <= 1.0


@patch("yfinance.Ticker")
async def test_game_plan_agent_structured_output_fallback(
    mock_yf_ticker, test_container, mock_market_fetcher
):
    """Test fallback when structured output fails."""
    import pandas as pd

    agent = test_container.game_plan_agent()
    agent.market_fetcher = mock_market_fetcher
    agent.llm.astructured = AsyncMock(side_effect=StructuredOutputError("Schema mismatch"))
    agent.llm.acomplete = AsyncMock(return_value="Market neutral, focus on tech")

    mock_ticker = MagicMock()
    mock_history_df = pd.DataFrame({"Close": [100.0, 102.0]})
    mock_ticker.history.return_value = mock_history_df
    mock_yf_ticker.return_value = mock_ticker

    plan = await agent.generate(["AAPL", "TSLA"])

    assert isinstance(plan, GamePlan)
    assert plan.risk_stance == "NEUTRAL"
    assert plan.confidence == 0.5


def test_game_plan_persist(test_container, sample_game_plan, tmp_path):
    """Test game plan persistence."""
    agent = test_container.game_plan_agent()
    plan_path = agent.persist(sample_game_plan, str(tmp_path))

    assert plan_path.exists()
    assert plan_path.name == f"{sample_game_plan.date}.json"


def test_fetch_futures_context(test_container, mock_market_fetcher):
    """Test futures context fetching."""
    agent = test_container.game_plan_agent()
    agent.market_fetcher = mock_market_fetcher
    result = agent._fetch_futures_context(["ES=F", "NQ=F"])

    assert result == {"ES=F": 0.5, "NQ=F": -0.3}


def test_fetch_futures_context_graceful(test_container):
    """Test futures unavailable graceful degradation."""
    agent = test_container.game_plan_agent()
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_overnight_futures.side_effect = ValueError("No data")
    agent.market_fetcher = mock_fetcher

    result = agent._fetch_futures_context(["ES=F"])

    assert result == {}


async def test_empty_watchlist_uses_defaults(test_container, mock_market_fetcher):
    """Test empty watchlist uses defaults."""
    agent = test_container.game_plan_agent()
    agent.market_fetcher = mock_market_fetcher
    agent.llm.astructured = AsyncMock(
        return_value=GamePlanLLMResponse(
            priority_symbols=["SPY"],
            risk_stance="NEUTRAL",
            sector_focus=["Market"],
            key_levels=[],
            reasoning="Default watchlist",
            confidence=0.6,
        )
    )

    plan = await agent.generate([])

    assert isinstance(plan, GamePlan)


def test_format_futures(test_container):
    """Test futures formatting."""
    agent = test_container.game_plan_agent()

    result = agent._format_futures({"ES=F": 0.5, "NQ=F": -0.3})
    assert "ES=F: +0.50% (up)" in result
    assert "NQ=F: -0.30% (down)" in result

    result = agent._format_futures({})
    assert result == "Futures data unavailable"


def test_format_overnight_summary(test_container):
    """Test overnight summary formatting."""
    agent = test_container.game_plan_agent()

    result = agent._format_overnight_summary({"ES=F": 0.5}, "AAPL +2.0%")
    assert "Futures:" in result
    assert "Pre-market:" in result
    assert "AAPL +2.0%" in result
