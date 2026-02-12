"""CPU profiler for daemon cycles using yappi."""

import contextlib
import time as time_mod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from src.daemon.profiling.metrics import FunctionStats, ProfilingMetrics
from src.daemon.profiling.storage import ProfileStorage


@asynccontextmanager
async def async_nullcontext() -> AsyncIterator[None]:
    """Async no-op context manager for when profiler is disabled.

    Yields:
        None
    """
    yield None


class CycleProfiler:
    """Manage per-cycle CPU profiling with yappi."""

    def __init__(
        self,
        storage: ProfileStorage,
        clock_type: str = "wall",
        top_n_functions: int = 50,
        sample_rate: int = 1,
    ) -> None:
        """Initialize cycle profiler.

        Args:
            storage: Profile storage manager
            clock_type: Clock type ("wall" or "cpu")
            top_n_functions: Number of top functions to track
            sample_rate: Sample every Nth cycle (1 = every cycle)
        """
        self.storage = storage
        self.clock_type = clock_type
        self.top_n_functions = top_n_functions
        self.sample_rate = sample_rate
        self._current_cycle = 0
        self._yappi_available = self._check_yappi()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CycleProfiler(clock={self.clock_type}, top_n={self.top_n_functions})"

    def _check_yappi(self) -> bool:
        """Check if yappi is available."""
        try:
            import yappi  # noqa: F401

            return True
        except ImportError:
            logger.opt(exception=True).warning("yappi not installed, profiling disabled")
            return False

    @contextlib.asynccontextmanager
    async def profile_cycle(self, cycle_num: int) -> AsyncIterator[ProfilingMetrics | None]:
        """Profile single daemon cycle.

        Args:
            cycle_num: Cycle number

        Yields:
            ProfilingMetrics or None if profiling disabled/skipped
        """
        self._current_cycle = cycle_num

        # Skip if yappi not available
        if not self._yappi_available:
            yield None
            return

        # Sample rate filtering
        if cycle_num % self.sample_rate != 0:
            yield None
            return

        import yappi

        # Start profiling
        yappi.set_clock_type(self.clock_type)
        yappi.start()
        start_time = time_mod.perf_counter()

        try:
            # Run cycle (yield None initially, metrics filled later)
            metrics_placeholder = None
            yield metrics_placeholder
        finally:
            # Stop profiling and calculate overhead
            yappi.stop()
            end_time = time_mod.perf_counter()
            duration = end_time - start_time

            # Calculate overhead
            overhead_start = time_mod.perf_counter()
            try:
                metrics = self._process_profile(cycle_num, duration)
                overhead = time_mod.perf_counter() - overhead_start
                metrics.profiling_overhead_percent = (overhead / duration * 100) if duration > 0 else 0.0

                # Warn if overhead exceeds threshold
                if metrics.profiling_overhead_percent > 5.0:
                    logger.warning(
                        f"Profiling overhead {metrics.profiling_overhead_percent:.1f}% "
                        f"exceeds 5% threshold - consider increasing sample_rate"
                    )
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to process profile: {e}")
                metrics = None
            finally:
                yappi.clear_stats()

            # Update placeholder (though caller already got None)
            # This is mainly for logging/debugging
            if metrics:
                logger.debug(f"Profiling complete: {metrics}")

    def _process_profile(self, cycle_num: int, duration: float) -> ProfilingMetrics:
        """Process yappi stats and save files.

        Args:
            cycle_num: Cycle number
            duration: Cycle duration in seconds

        Returns:
            ProfilingMetrics with aggregated stats
        """
        import yappi

        stats = yappi.get_func_stats()
        timestamp = datetime.now(UTC)

        # Extract top functions
        top_functions = self._extract_top_functions(stats)

        # Calculate percentiles
        all_cumtimes = [s.ttot for s in stats]
        p50, p95, p99 = self._calculate_percentiles(all_cumtimes)

        # Create metrics
        metrics = ProfilingMetrics(
            cycle_number=cycle_num,
            timestamp=timestamp,
            duration_seconds=duration,
            profiling_overhead_percent=0.0,  # Filled by caller
            top_functions=top_functions,
            p50_function_time=p50,
            p95_function_time=p95,
            p99_function_time=p99,
        )

        # Save profiles
        try:
            self._save_profiles(cycle_num, stats, metrics, timestamp)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to save profiles: {e}")

        return metrics

    def _extract_top_functions(self, stats: object) -> list[FunctionStats]:
        """Extract top N functions from yappi stats.

        Args:
            stats: yappi function stats

        Returns:
            List of FunctionStats
        """
        top_funcs = []
        for stat in stats[: self.top_n_functions]:  # type: ignore[index]
            function_name = f"{stat.module}:{stat.name}"
            top_funcs.append(
                FunctionStats(
                    function=function_name,
                    cumtime=stat.ttot,
                    ncalls=stat.ncall,
                    percall=stat.ttot / stat.ncall if stat.ncall > 0 else 0.0,
                )
            )
        return top_funcs

    def _calculate_percentiles(self, values: list[float]) -> tuple[float | None, float | None, float | None]:
        """Calculate p50, p95, p99 percentiles.

        Args:
            values: List of values

        Returns:
            Tuple of (p50, p95, p99)
        """
        if not values:
            return None, None, None

        sorted_values = sorted(values)
        n = len(sorted_values)

        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        return (
            sorted_values[p50_idx] if p50_idx < n else None,
            sorted_values[p95_idx] if p95_idx < n else None,
            sorted_values[p99_idx] if p99_idx < n else None,
        )

    def _save_profiles(
        self,
        cycle_num: int,
        stats: object,
        metrics: ProfilingMetrics,
        timestamp: datetime,
    ) -> None:
        """Save profile files (pstats and JSON).

        Args:
            cycle_num: Cycle number
            stats: yappi function stats
            metrics: Profiling metrics
            timestamp: Profile timestamp
        """
        # Save pstats (yappi requires file path, not BytesIO)
        import tempfile

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pstats") as tmp:
            tmp_path = Path(tmp.name)
        try:
            from typing import cast

            import yappi

            yappi_stats = cast(yappi.YFuncStats, stats)
            yappi_stats.save(str(tmp_path), type="pstat")
            pstats_data = tmp_path.read_bytes()
            self.storage.save_pstats(cycle_num, pstats_data, timestamp)
        finally:
            tmp_path.unlink()

        # Save JSON summary
        json_data = {
            "cycle_number": cycle_num,
            "timestamp": timestamp.isoformat(),
            "duration_seconds": metrics.duration_seconds,
            "profiling_overhead_percent": metrics.profiling_overhead_percent,
            "top_functions": [
                {
                    "function": f.function,
                    "cumtime": f.cumtime,
                    "ncalls": f.ncalls,
                    "percall": f.percall,
                }
                for f in metrics.top_functions
            ],
            "p50_function_time": metrics.p50_function_time,
            "p95_function_time": metrics.p95_function_time,
            "p99_function_time": metrics.p99_function_time,
        }
        self.storage.save_json(cycle_num, json_data, timestamp)

        # Cleanup old profiles
        self.storage.cleanup()
