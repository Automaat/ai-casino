"""Profile storage and rotation management."""

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger


class ProfileStorage:
    """Manage profile file storage and rotation."""

    def __init__(
        self,
        output_dir: str,
        retention_days: int = 7,
        max_files: int = 1000,
        max_disk_mb: int = 500,
    ) -> None:
        """Initialize profile storage.

        Args:
            output_dir: Base directory for profiles (supports ~ expansion)
            retention_days: Days to retain profiles
            max_files: Maximum number of profile files
            max_disk_mb: Maximum disk usage in MB
        """
        self.output_dir = Path(output_dir).expanduser()
        self.retention_days = retention_days
        self.max_files = max_files
        self.max_disk_mb = max_disk_mb
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ProfileStorage(dir={self.output_dir}, retention={self.retention_days}d)"

    def get_cycle_dir(self, date: datetime | None = None) -> Path:
        """Get directory for cycle profiles by date.

        Args:
            date: Date for directory (defaults to today)

        Returns:
            Path to cycle directory
        """
        if date is None:
            date = datetime.now(UTC)
        date_str = date.strftime("%Y-%m-%d")
        cycle_dir = self.output_dir / date_str
        cycle_dir.mkdir(parents=True, exist_ok=True)
        return cycle_dir

    def save_pstats(self, cycle_num: int, stats_data: bytes, timestamp: datetime) -> Path:
        """Save pstats binary file.

        Args:
            cycle_num: Cycle number
            stats_data: Binary pstats data
            timestamp: Profile timestamp

        Returns:
            Path to saved file
        """
        cycle_dir = self.get_cycle_dir(timestamp)
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"cycle_{cycle_num:04d}_{timestamp_str}.pstats"
        filepath = cycle_dir / filename

        try:
            filepath.write_bytes(stats_data)
            logger.debug(f"Saved pstats: {filepath}")
            return filepath
        except OSError as e:
            logger.error(f"Failed to save pstats: {e}")
            raise

    def save_json(self, cycle_num: int, metrics: dict, timestamp: datetime) -> Path:
        """Save JSON metrics summary.

        Args:
            cycle_num: Cycle number
            metrics: Metrics dictionary
            timestamp: Profile timestamp

        Returns:
            Path to saved file
        """
        cycle_dir = self.get_cycle_dir(timestamp)
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"cycle_{cycle_num:04d}_{timestamp_str}.json"
        filepath = cycle_dir / filename

        try:
            with filepath.open("w") as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.debug(f"Saved JSON: {filepath}")
            return filepath
        except OSError as e:
            logger.error(f"Failed to save JSON: {e}")
            raise

    def cleanup(self) -> None:
        """Cleanup old profiles based on retention policy."""
        try:
            self._cleanup_by_age()
            self._cleanup_by_count()
            self._cleanup_by_disk_usage()
        except Exception as e:
            logger.opt(exception=True).warning(f"Cleanup failed: {e}")

    def _cleanup_by_age(self) -> None:
        """Remove profiles older than retention_days."""
        cutoff = datetime.now(UTC) - timedelta(days=self.retention_days)
        removed = 0

        for date_dir in self.output_dir.iterdir():
            if not date_dir.is_dir():
                continue

            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").replace(tzinfo=UTC)
                if dir_date < cutoff:
                    shutil.rmtree(date_dir)
                    removed += 1
                    logger.debug(f"Removed old profile dir: {date_dir.name}")
            except ValueError:
                continue

        if removed > 0:
            logger.info(f"Cleaned up {removed} old profile directories")

    def _cleanup_by_count(self) -> None:
        """Remove oldest files if exceeding max_files."""
        all_files = sorted(
            self.output_dir.rglob("cycle_*.pstats"),
            key=lambda p: p.stat().st_mtime,
        )

        if len(all_files) > self.max_files:
            excess = len(all_files) - self.max_files
            for filepath in all_files[:excess]:
                try:
                    filepath.unlink()
                    json_path = filepath.with_suffix(".json")
                    if json_path.exists():
                        json_path.unlink()
                    logger.debug(f"Removed excess file: {filepath.name}")
                except OSError as e:
                    logger.warning(f"Failed to remove {filepath}: {e}")

            logger.info(f"Cleaned up {excess} excess profile files")

    def _cleanup_by_disk_usage(self) -> None:
        """Remove oldest files if exceeding max_disk_mb."""
        total_size_mb = sum(f.stat().st_size for f in self.output_dir.rglob("*") if f.is_file()) / (
            1024 * 1024
        )

        if total_size_mb <= self.max_disk_mb:
            return

        all_files = sorted(
            self.output_dir.rglob("cycle_*.*"),
            key=lambda p: p.stat().st_mtime,
        )

        removed_mb = 0.0
        for filepath in all_files:
            if total_size_mb - removed_mb <= self.max_disk_mb:
                break

            try:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                filepath.unlink()
                removed_mb += size_mb
                logger.debug(f"Removed file to free disk: {filepath.name}")
            except OSError as e:
                logger.warning(f"Failed to remove {filepath}: {e}")

        if removed_mb > 0:
            logger.info(f"Freed {removed_mb:.1f}MB of disk space")
