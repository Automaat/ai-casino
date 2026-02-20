"""Tests for game plan agent."""

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.game_plan import GamePlan, GamePlanLLMResponse, KeyLevel
from src.models.providers.base import StructuredOutputError


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


async def test_game_plan_agent_generate(test_container, mock_market_fetcher):
    """Test game plan generation."""
    agent = test_container.game_plan_agent()
    agent.market_fetcher = mock_market_fetcher
    agent.llm.acomplete_with_tools = AsyncMock(return_value="Market research context")
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

    plan = await agent.generate(["AAPL", "TSLA", "GOOGL"])

    assert isinstance(plan, GamePlan)
    assert plan.priority_symbols == ["AAPL", "TSLA"]
    assert plan.risk_stance == "NEUTRAL"
    assert plan.key_levels == {"AAPL": 175.0}
    assert 0.0 <= plan.confidence <= 1.0


async def test_game_plan_agent_structured_output_fallback(test_container, mock_market_fetcher):
    """Test fallback when structured output fails."""
    agent = test_container.game_plan_agent()
    agent.market_fetcher = mock_market_fetcher
    agent.llm.acomplete_with_tools = AsyncMock(return_value="Market research context")
    agent.llm.astructured = AsyncMock(side_effect=StructuredOutputError("Schema mismatch"))
    agent.llm.acomplete = AsyncMock(return_value="Market neutral, focus on tech")

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


async def test_empty_watchlist_uses_defaults(test_container, mock_market_fetcher):
    """Test empty watchlist uses defaults."""
    agent = test_container.game_plan_agent()
    agent.market_fetcher = mock_market_fetcher
    agent.llm.acomplete_with_tools = AsyncMock(return_value="Market research context")
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
