"""Tests for FetchSectorPerformanceTool."""

from unittest.mock import patch

import pytest

from src.tools.game_plan.sector_performance import SECTOR_ETFS, FetchSectorPerformanceTool


class TestFetchSectorPerformanceTool:
    """Tests for FetchSectorPerformanceTool."""

    @pytest.mark.unit
    def test_sector_etfs_defined(self) -> None:
        """Sector ETF mapping is complete."""
        assert len(SECTOR_ETFS) == 10
        assert "XLK" in SECTOR_ETFS
        assert "XLE" in SECTOR_ETFS

    @pytest.mark.unit
    def test_tool_definition(self) -> None:
        """Tool definition has correct name."""
        tool = FetchSectorPerformanceTool()
        defn = tool.get_tool_definition()

        assert defn.function.name == "fetch_sector_performance"

    @pytest.mark.unit
    def test_handles_yfinance_import_error(self) -> None:
        """Graceful fallback when yfinance unavailable."""
        tool = FetchSectorPerformanceTool()

        with patch.dict("sys.modules", {"yfinance": None}):
            with patch("builtins.__import__", side_effect=ImportError("no yfinance")):
                result = tool.execute()

        assert "not available" in result.lower() or "unavailable" in result.lower()
