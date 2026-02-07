"""Tests for game plan agent."""

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.game_plan import GamePlan, GamePlanAgent, GamePlanLLMResponse
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
def mock_llm_client():
    """Mock LLM client."""
    client = MagicMock()
    client.astructured = AsyncMock()
    client.acomplete = AsyncMock()
    return client


@pytest.fixture
def mock_market_fetcher():
    """Mock market data fetcher."""
    fetcher = MagicMock()
    fetcher.fetch_overnight_futures = MagicMock(return_value={"ES=F": 0.5, "NQ=F": -0.3})
    return fetcher


@pytest.mark.asyncio
async def test_game_plan_agent_generate(mock_llm_client, mock_market_fetcher):
    """Test game plan generation."""
    mock_llm_client.astructured.return_value = GamePlanLLMResponse(
        priority_symbols=["AAPL", "TSLA"],
        risk_stance="NEUTRAL",
        sector_focus=["Technology"],
        key_levels={"AAPL": 175.0},
        reasoning="Tech strength overnight",
        confidence=0.8,
    )

    agent = GamePlanAgent(mock_llm_client, mock_market_fetcher)
    plan = await agent.generate(["AAPL", "TSLA", "GOOGL"])

    assert isinstance(plan, GamePlan)
    assert plan.priority_symbols == ["AAPL", "TSLA"]
    assert plan.risk_stance == "NEUTRAL"
    assert 0.0 <= plan.confidence <= 1.0
    mock_llm_client.astructured.assert_called_once()


@pytest.mark.asyncio
async def test_game_plan_agent_structured_output_fallback(mock_llm_client, mock_market_fetcher):
    """Test fallback when structured output fails."""
    mock_llm_client.astructured.side_effect = StructuredOutputError("Schema mismatch")
    mock_llm_client.acomplete.return_value = "Market neutral, focus on tech"

    agent = GamePlanAgent(mock_llm_client, mock_market_fetcher)
    plan = await agent.generate(["AAPL", "TSLA"])

    assert isinstance(plan, GamePlan)
    assert plan.risk_stance == "NEUTRAL"
    assert plan.confidence == 0.5
    mock_llm_client.acomplete.assert_called_once()


def test_game_plan_persist(sample_game_plan, tmp_path):
    """Test game plan persistence."""
    agent = GamePlanAgent(MagicMock(), MagicMock())
    plan_path = agent.persist(sample_game_plan, str(tmp_path))

    assert plan_path.exists()
    assert plan_path.name == f"{sample_game_plan.date}.json"


def test_fetch_futures_context(mock_market_fetcher):
    """Test futures context fetching."""
    agent = GamePlanAgent(MagicMock(), mock_market_fetcher)
    result = agent._fetch_futures_context(["ES=F", "NQ=F"])

    assert result == {"ES=F": 0.5, "NQ=F": -0.3}


def test_fetch_futures_context_graceful(mock_market_fetcher):
    """Test futures unavailable graceful degradation."""
    mock_market_fetcher.fetch_overnight_futures.side_effect = ValueError("No data")

    agent = GamePlanAgent(MagicMock(), mock_market_fetcher)
    result = agent._fetch_futures_context(["ES=F"])

    assert result == {}


@pytest.mark.asyncio
async def test_empty_watchlist_uses_defaults(mock_llm_client, mock_market_fetcher):
    """Test empty watchlist uses defaults."""
    mock_llm_client.astructured.return_value = GamePlanLLMResponse(
        priority_symbols=["SPY"],
        risk_stance="NEUTRAL",
        sector_focus=["Market"],
        key_levels={},
        reasoning="Default watchlist",
        confidence=0.6,
    )

    agent = GamePlanAgent(mock_llm_client, mock_market_fetcher)
    plan = await agent.generate([])

    assert isinstance(plan, GamePlan)
    mock_llm_client.astructured.assert_called_once()


def test_format_futures():
    """Test futures formatting."""
    agent = GamePlanAgent(MagicMock(), MagicMock())

    result = agent._format_futures({"ES=F": 0.5, "NQ=F": -0.3})
    assert "ES=F: +0.50% (up)" in result
    assert "NQ=F: -0.30% (down)" in result

    result = agent._format_futures({})
    assert result == "Futures data unavailable"


def test_format_overnight_summary():
    """Test overnight summary formatting."""
    agent = GamePlanAgent(MagicMock(), MagicMock())

    result = agent._format_overnight_summary({"ES=F": 0.5}, "AAPL +2.0%")
    assert "Futures:" in result
    assert "Pre-market:" in result
    assert "AAPL +2.0%" in result
