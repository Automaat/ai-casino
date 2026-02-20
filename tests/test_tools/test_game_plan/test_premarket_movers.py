"""Tests for FetchPremarketMoversTool."""

from unittest.mock import patch

import pytest

from src.tools.game_plan.premarket_movers import FetchPremarketMoversTool


class TestFetchPremarketMoversTool:
    """Tests for FetchPremarketMoversTool."""

    @pytest.mark.unit
    def test_formats_movers(self) -> None:
        """Formats movers when data available."""
        tool = FetchPremarketMoversTool()

        with patch.object(tool, "_scan_movers", return_value=[("AAPL", 2.5), ("TSLA", -1.3)]):
            result = tool.execute(symbols="AAPL,TSLA")

        assert "AAPL" in result
        assert "+2.5%" in result
        assert "TSLA" in result
        assert "-1.3%" in result

    @pytest.mark.unit
    def test_handles_no_data(self) -> None:
        """Returns message when no movers found."""
        tool = FetchPremarketMoversTool()

        with patch.object(tool, "_scan_movers", return_value=[]):
            result = tool.execute(symbols="XYZ")

        assert "No pre-market data" in result

    @pytest.mark.unit
    def test_limits_to_15_symbols(self) -> None:
        """Limits scan to 15 symbols."""
        tool = FetchPremarketMoversTool()
        symbols = ",".join([f"SYM{i}" for i in range(20)])

        with patch.object(tool, "_scan_movers", return_value=[]) as mock_scan:
            tool.execute(symbols=symbols)

        assert len(mock_scan.call_args[0][0]) == 15

    @pytest.mark.unit
    def test_tool_definition(self) -> None:
        """Tool definition has correct name."""
        tool = FetchPremarketMoversTool()
        defn = tool.get_tool_definition()

        assert defn.function.name == "fetch_premarket_movers"
