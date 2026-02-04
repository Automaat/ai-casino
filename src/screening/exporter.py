"""Export screening results and manage watchlists."""

import csv
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from src.screening.screener import ScreeningCriteria, ScreeningOutput, ScreeningResult


class ExportFormat(StrEnum):
    """Export file format."""

    CSV = "csv"
    JSON = "json"


class WatchlistEntry(BaseModel):
    """Single watchlist entry."""

    symbol: str
    name: str
    added_at: datetime
    criteria: ScreeningCriteria
    score: float
    notes: str | None = None


class Watchlist(BaseModel):
    """Watchlist container."""

    name: str
    entries: list[WatchlistEntry]
    created_at: datetime
    updated_at: datetime


def _get_default_export_dir() -> Path:
    """Get default export directory.

    Returns:
        Path to ~/.ai-casino/exports/
    """
    return Path.home() / ".ai-casino" / "exports"


def _get_default_watchlist_dir() -> Path:
    """Get default watchlist directory.

    Returns:
        Path to ~/.ai-casino/watchlists/
    """
    return Path.home() / ".ai-casino" / "watchlists"


class ScreeningExporter:
    """Export screening results and manage watchlists."""

    def __init__(
        self,
        export_dir: Path | None = None,
        watchlist_dir: Path | None = None,
    ) -> None:
        """Initialize screening exporter.

        Args:
            export_dir: Directory for exports. Defaults to ~/.ai-casino/exports/
            watchlist_dir: Directory for watchlists. Defaults to ~/.ai-casino/watchlists/
        """
        self._export_dir = export_dir or _get_default_export_dir()
        self._watchlist_dir = watchlist_dir or _get_default_watchlist_dir()

        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._watchlist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ScreeningExporter (export_dir={self._export_dir})")

    def export_to_csv(
        self,
        output: ScreeningOutput,
        filename: str | None = None,
    ) -> Path:
        """Export screening results to CSV.

        Args:
            output: Screening output to export
            filename: Optional filename (without extension)

        Returns:
            Path to created CSV file
        """
        if not filename:
            timestamp = output.screened_at.strftime("%Y%m%d_%H%M%S")
            filename = f"{output.criteria.value}_{output.universe}_{timestamp}"

        filepath = self._export_dir / f"{filename}.csv"

        with filepath.open("w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "symbol",
                    "name",
                    "sector",
                    "score",
                    "signal",
                    "reason",
                    "screened_at",
                    "criteria",
                    "universe",
                    *self._get_metric_columns(output.results),
                ]
            )

            # Data rows
            for result in output.results:
                row = [
                    result.symbol,
                    result.name,
                    result.sector,
                    result.score,
                    result.signal.value,
                    result.reason,
                    output.screened_at.isoformat(),
                    output.criteria.value,
                    output.universe,
                ]
                # Add metric values in consistent order
                for col in self._get_metric_columns(output.results):
                    row.append(result.metrics.get(col, ""))
                writer.writerow(row)

        logger.info(f"Exported {len(output.results)} results to {filepath}")
        return filepath

    def export_to_json(
        self,
        output: ScreeningOutput,
        filename: str | None = None,
    ) -> Path:
        """Export screening results to JSON.

        Args:
            output: Screening output to export
            filename: Optional filename (without extension)

        Returns:
            Path to created JSON file
        """
        if not filename:
            timestamp = output.screened_at.strftime("%Y%m%d_%H%M%S")
            filename = f"{output.criteria.value}_{output.universe}_{timestamp}"

        filepath = self._export_dir / f"{filename}.json"

        data = output.model_dump()
        data["screened_at"] = output.screened_at.isoformat()

        with filepath.open("w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(output.results)} results to {filepath}")
        return filepath

    def save_to_watchlist(
        self,
        results: list[ScreeningResult],
        criteria: ScreeningCriteria,
        watchlist_name: str = "default",
        notes: str | None = None,
    ) -> Watchlist:
        """Save screening results to watchlist.

        Args:
            results: Results to add to watchlist
            criteria: Screening criteria used
            watchlist_name: Watchlist name
            notes: Optional notes for entries

        Returns:
            Updated Watchlist
        """
        watchlist = self.load_watchlist(watchlist_name)
        now = datetime.now(UTC)

        if not watchlist:
            watchlist = Watchlist(
                name=watchlist_name,
                entries=[],
                created_at=now,
                updated_at=now,
            )

        # Add new entries (avoid duplicates by symbol)
        existing_symbols = {e.symbol for e in watchlist.entries}
        for result in results:
            if result.symbol not in existing_symbols:
                entry = WatchlistEntry(
                    symbol=result.symbol,
                    name=result.name,
                    added_at=now,
                    criteria=criteria,
                    score=result.score,
                    notes=notes,
                )
                watchlist.entries.append(entry)
                existing_symbols.add(result.symbol)

        watchlist.updated_at = now

        # Save to disk
        filepath = self._watchlist_dir / f"{watchlist_name}.json"
        with filepath.open("w") as f:
            json.dump(watchlist.model_dump(), f, indent=2, default=str)

        logger.info(f"Saved {len(results)} entries to watchlist '{watchlist_name}'")
        return watchlist

    def load_watchlist(self, name: str = "default") -> Watchlist | None:
        """Load watchlist by name.

        Args:
            name: Watchlist name

        Returns:
            Watchlist or None if not found
        """
        filepath = self._watchlist_dir / f"{name}.json"

        if not filepath.exists():
            return None

        with filepath.open() as f:
            data = json.load(f)

        # Parse datetime strings
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        for entry in data["entries"]:
            entry["added_at"] = datetime.fromisoformat(entry["added_at"])

        return Watchlist.model_validate(data)

    def list_watchlists(self) -> list[str]:
        """List all available watchlists.

        Returns:
            List of watchlist names
        """
        watchlists = []
        for filepath in self._watchlist_dir.glob("*.json"):
            watchlists.append(filepath.stem)
        return sorted(watchlists)

    def delete_watchlist(self, name: str) -> bool:
        """Delete a watchlist.

        Args:
            name: Watchlist name

        Returns:
            True if deleted, False if not found
        """
        filepath = self._watchlist_dir / f"{name}.json"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted watchlist '{name}'")
            return True
        return False

    def remove_from_watchlist(self, symbol: str, watchlist_name: str = "default") -> bool:
        """Remove a symbol from watchlist.

        Args:
            symbol: Stock symbol to remove
            watchlist_name: Watchlist name

        Returns:
            True if removed, False if not found
        """
        watchlist = self.load_watchlist(watchlist_name)
        if not watchlist:
            return False

        original_count = len(watchlist.entries)
        watchlist.entries = [e for e in watchlist.entries if e.symbol != symbol]

        if len(watchlist.entries) < original_count:
            watchlist.updated_at = datetime.now(UTC)
            filepath = self._watchlist_dir / f"{watchlist_name}.json"
            with filepath.open("w") as f:
                json.dump(watchlist.model_dump(), f, indent=2, default=str)
            logger.info(f"Removed {symbol} from watchlist '{watchlist_name}'")
            return True

        return False

    def _get_metric_columns(self, results: list[ScreeningResult]) -> list[str]:
        """Get all metric column names from results.

        Args:
            results: Screening results

        Returns:
            Sorted list of metric column names
        """
        columns = set()
        for result in results:
            columns.update(result.metrics.keys())
        return sorted(columns)

    def __repr__(self) -> str:
        """String representation."""
        return f"ScreeningExporter(export_dir={self._export_dir})"
