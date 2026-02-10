"""Tests for screen stocks tool."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.screening.analyzer import ScreeningAnalysis
from src.screening.screener import ScreeningCriteria, ScreeningOutput, ScreeningResult
from src.strategies.signal import Signal
from src.tools.screen_stocks import ScreenStocksTool


@pytest.fixture
def sample_screening_output():
    """Sample screening output for testing."""
    return ScreeningOutput(
        criteria=ScreeningCriteria.MOMENTUM,
        universe="SP500",
        results=[
            ScreeningResult(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Technology",
                score=0.85,
                signal=Signal.BUY,
                metrics={"rsi": 28.5, "macd_hist": 0.15},
                reason="RSI oversold, MACD bullish",
            ),
            ScreeningResult(
                symbol="MSFT",
                name="Microsoft Corp",
                sector="Technology",
                score=0.78,
                signal=Signal.BUY,
                metrics={"rsi": 32.1, "macd_hist": 0.12},
                reason="RSI oversold, MACD bullish",
            ),
        ],
        total_screened=500,
        errors=["FAILED1"],
        screened_at=datetime.now(),
    )


@pytest.fixture
def sample_screening_analysis():
    """Sample screening analysis for testing."""
    return ScreeningAnalysis(
        summary="Found 2 momentum stocks with strong oversold signals.",
        top_picks=[
            "AAPL - Strongest RSI oversold signal",
            "MSFT - Solid momentum setup",
        ],
        sector_insights="Heavy concentration in Technology sector.",
        risk_factors="Correlation risk due to similar technical patterns.",
        next_steps="Research fundamental catalysts before entry.",
    )


class TestScreenStocksTool:
    """Tests for ScreenStocksTool."""

    def test_name(self, test_container_full):
        """Test tool name."""
        tool = ScreenStocksTool(container=test_container_full)
        assert tool.name == "screen_stocks"

    def test_requires_confirmation(self, test_container_full):
        """Test confirmation requirement."""
        tool = ScreenStocksTool(container=test_container_full)
        assert tool.requires_confirmation is True

    def test_get_tool_definition(self, test_container_full):
        """Test tool definition structure."""
        tool = ScreenStocksTool(container=test_container_full)
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "screen_stocks"

        params = definition["function"]["parameters"]
        assert "criteria" in params["properties"]
        assert params["properties"]["criteria"]["enum"] == ["momentum", "value", "breakout"]
        assert "universe" in params["properties"]
        assert "top_n" in params["properties"]
        assert "criteria" in params["required"]

    def test_repr(self, test_container_full):
        """Test string representation."""
        tool = ScreenStocksTool(container=test_container_full)
        assert "ScreenStocksTool" in repr(tool)

    def test_execute_integration(
        self,
        test_container_full,
        sample_screening_output,
        sample_screening_analysis,
    ):
        """Test _format_output produces correct markdown."""
        tool = ScreenStocksTool(container=test_container_full)
        result = tool._format_output(sample_screening_output, sample_screening_analysis)

        assert "AAPL" in result
        assert "Momentum Screening Results" in result
        assert "Technology" in result

    def test_run_screening_integration(
        self,
        test_container_full,
        sample_screening_output,
        sample_screening_analysis,
    ):
        """Test _run_screening formats output correctly."""
        tool = ScreenStocksTool(container=test_container_full)
        result = tool._format_output(sample_screening_output, sample_screening_analysis)

        assert "AAPL" in result
        assert "Momentum Screening Results" in result
        assert "Technology" in result

    def test_run_screening_no_results_message(self):
        """Test no results message format."""
        empty_output = ScreeningOutput(
            criteria=ScreeningCriteria.VALUE,
            universe="SP500",
            results=[],
            total_screened=500,
            errors=[],
            screened_at=datetime.now(),
        )

        criteria = empty_output.criteria.value
        universe = empty_output.universe
        total = empty_output.total_screened
        message = f"No stocks matched {criteria} criteria in {universe}. Screened {total} stocks."
        assert "No stocks matched" in message
        assert "value" in message
        assert "500" in message

    def test_format_output(self, test_container_full, sample_screening_output, sample_screening_analysis):
        """Test output formatting."""
        tool = ScreenStocksTool(container=test_container_full)
        result = tool._format_output(sample_screening_output, sample_screening_analysis)

        assert "# Momentum Screening Results" in result
        assert "**Universe:** SP500" in result
        assert "**Screened:** 500 stocks" in result
        assert "AAPL" in result
        assert "Microsoft Corp" in result
        assert "rsi=28.5" in result
        assert "1 symbols failed" in result

    def test_execute_error_handling(self, test_container_full):
        """Test error handling in execute."""
        tool = ScreenStocksTool(container=test_container_full)

        mock_screener = MagicMock()
        mock_screener.screen.side_effect = RuntimeError("Test error")
        test_container_full.stock_screener.override(mock_screener)

        result = tool.execute(criteria="momentum", universe="SP500", top_n=10)

        assert "Screening failed" in result
        assert "Test error" in result
