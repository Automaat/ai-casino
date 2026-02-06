"""Tests for screening exporter."""

import csv
import json
from datetime import datetime

import pytest

from src.screening.exporter import (
    ExportFormat,
    ScreeningExporter,
    Watchlist,
    WatchlistEntry,
)
from src.screening.screener import ScreeningCriteria, ScreeningOutput, ScreeningResult
from src.strategies.signal import Signal


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
        screened_at=datetime(2024, 1, 15, 10, 0, 0),
    )


@pytest.fixture
def exporter(tmp_path):
    """Create ScreeningExporter with temp directories."""
    return ScreeningExporter(
        export_dir=tmp_path / "exports",
        watchlist_dir=tmp_path / "watchlists",
    )


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_values(self):
        """Test format values."""
        assert ExportFormat.CSV == "csv"
        assert ExportFormat.JSON == "json"


class TestWatchlistEntry:
    """Tests for WatchlistEntry model."""

    def test_create(self):
        """Test WatchlistEntry creation."""
        entry = WatchlistEntry(
            symbol="AAPL",
            name="Apple Inc.",
            added_at=datetime.now(),
            criteria=ScreeningCriteria.MOMENTUM,
            score=0.85,
            notes="Test note",
        )

        assert entry.symbol == "AAPL"
        assert entry.criteria == ScreeningCriteria.MOMENTUM
        assert entry.notes == "Test note"

    def test_create_without_notes(self):
        """Test WatchlistEntry without notes."""
        entry = WatchlistEntry(
            symbol="AAPL",
            name="Apple Inc.",
            added_at=datetime.now(),
            criteria=ScreeningCriteria.VALUE,
            score=0.75,
        )

        assert entry.notes is None


class TestWatchlist:
    """Tests for Watchlist model."""

    def test_create(self):
        """Test Watchlist creation."""
        now = datetime.now()
        watchlist = Watchlist(
            name="test",
            entries=[],
            created_at=now,
            updated_at=now,
        )

        assert watchlist.name == "test"
        assert len(watchlist.entries) == 0


class TestScreeningExporter:
    """Tests for ScreeningExporter."""

    def test_init(self, exporter):
        """Test exporter initialization."""
        assert "ScreeningExporter" in repr(exporter)

    def test_export_to_csv(self, exporter, sample_screening_output):
        """Test CSV export."""
        filepath = exporter.export_to_csv(sample_screening_output)

        assert filepath.exists()
        assert filepath.suffix == ".csv"

        with filepath.open() as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3  # Header + 2 results
        assert "symbol" in rows[0]
        assert "AAPL" in rows[1]

    def test_export_to_csv_custom_filename(self, exporter, sample_screening_output):
        """Test CSV export with custom filename."""
        filepath = exporter.export_to_csv(sample_screening_output, filename="custom_export")

        assert filepath.name == "custom_export.csv"

    def test_export_to_json(self, exporter, sample_screening_output):
        """Test JSON export."""
        filepath = exporter.export_to_json(sample_screening_output)

        assert filepath.exists()
        assert filepath.suffix == ".json"

        with filepath.open() as f:
            data = json.load(f)

        assert data["criteria"] == "momentum"
        assert len(data["results"]) == 2
        assert data["results"][0]["symbol"] == "AAPL"

    def test_export_to_json_custom_filename(self, exporter, sample_screening_output):
        """Test JSON export with custom filename."""
        filepath = exporter.export_to_json(sample_screening_output, filename="my_results")

        assert filepath.name == "my_results.json"

    def test_save_to_watchlist_new(self, exporter, sample_screening_output):
        """Test saving to new watchlist."""
        results = sample_screening_output.results

        watchlist = exporter.save_to_watchlist(
            results=results,
            criteria=ScreeningCriteria.MOMENTUM,
            watchlist_name="my_picks",
        )

        assert watchlist.name == "my_picks"
        assert len(watchlist.entries) == 2
        assert watchlist.entries[0].symbol == "AAPL"

    def test_save_to_watchlist_existing(self, exporter, sample_screening_output):
        """Test saving to existing watchlist doesn't duplicate."""
        results = sample_screening_output.results

        exporter.save_to_watchlist(
            results=results,
            criteria=ScreeningCriteria.MOMENTUM,
            watchlist_name="existing",
        )

        # Save again - should not duplicate
        watchlist = exporter.save_to_watchlist(
            results=results,
            criteria=ScreeningCriteria.MOMENTUM,
            watchlist_name="existing",
        )

        assert len(watchlist.entries) == 2  # Still 2, not 4

    def test_save_to_watchlist_with_notes(self, exporter, sample_screening_output):
        """Test saving with notes."""
        results = sample_screening_output.results[:1]

        watchlist = exporter.save_to_watchlist(
            results=results,
            criteria=ScreeningCriteria.MOMENTUM,
            watchlist_name="noted",
            notes="High conviction pick",
        )

        assert watchlist.entries[0].notes == "High conviction pick"

    def test_load_watchlist(self, exporter, sample_screening_output):
        """Test loading watchlist."""
        exporter.save_to_watchlist(
            results=sample_screening_output.results,
            criteria=ScreeningCriteria.MOMENTUM,
            watchlist_name="loadtest",
        )

        watchlist = exporter.load_watchlist("loadtest")

        assert watchlist is not None
        assert watchlist.name == "loadtest"
        assert len(watchlist.entries) == 2

    def test_load_watchlist_not_found(self, exporter):
        """Test loading non-existent watchlist."""
        watchlist = exporter.load_watchlist("nonexistent")

        assert watchlist is None

    def test_list_watchlists(self, exporter, sample_screening_output):
        """Test listing watchlists."""
        exporter.save_to_watchlist(
            results=sample_screening_output.results[:1],
            criteria=ScreeningCriteria.MOMENTUM,
            watchlist_name="list1",
        )
        exporter.save_to_watchlist(
            results=sample_screening_output.results[:1],
            criteria=ScreeningCriteria.VALUE,
            watchlist_name="list2",
        )

        watchlists = exporter.list_watchlists()

        assert "list1" in watchlists
        assert "list2" in watchlists
        assert len(watchlists) >= 2

    def test_delete_watchlist(self, exporter, sample_screening_output):
        """Test deleting watchlist."""
        exporter.save_to_watchlist(
            results=sample_screening_output.results[:1],
            criteria=ScreeningCriteria.MOMENTUM,
            watchlist_name="todelete",
        )

        result = exporter.delete_watchlist("todelete")
        assert result is True

        watchlist = exporter.load_watchlist("todelete")
        assert watchlist is None

    def test_delete_watchlist_not_found(self, exporter):
        """Test deleting non-existent watchlist."""
        result = exporter.delete_watchlist("nonexistent")
        assert result is False

    def test_remove_from_watchlist(self, exporter, sample_screening_output):
        """Test removing symbol from watchlist."""
        exporter.save_to_watchlist(
            results=sample_screening_output.results,
            criteria=ScreeningCriteria.MOMENTUM,
            watchlist_name="removefrom",
        )

        result = exporter.remove_from_watchlist("AAPL", "removefrom")
        assert result is True

        watchlist = exporter.load_watchlist("removefrom")
        assert len(watchlist.entries) == 1
        assert watchlist.entries[0].symbol == "MSFT"

    def test_remove_from_watchlist_not_found(self, exporter, sample_screening_output):
        """Test removing non-existent symbol."""
        exporter.save_to_watchlist(
            results=sample_screening_output.results[:1],
            criteria=ScreeningCriteria.MOMENTUM,
            watchlist_name="removetest",
        )

        result = exporter.remove_from_watchlist("INVALID", "removetest")
        assert result is False

    def test_get_metric_columns(self, exporter, sample_screening_output):
        """Test metric column extraction."""
        columns = exporter._get_metric_columns(sample_screening_output.results)

        assert "rsi" in columns
        assert "macd_hist" in columns
        assert columns == sorted(columns)  # Should be sorted
