"""API health checks and state cleanup for the trading daemon."""

import asyncio
import json
import os
import time
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import httpx
from loguru import logger
from pydantic import BaseModel

from src.daemon.config import DaemonConfig
from src.daemon.notifications import NotificationService
from src.daemon.state import DaemonState
from src.di.container import AppContainer


class ServiceStatus(StrEnum):
    """Health check status for a service."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    SKIPPED = "SKIPPED"


class ServiceCheckResult(BaseModel):
    """Result of a single service health check."""

    service: str
    status: ServiceStatus
    message: str
    duration_ms: float
    checked_at: datetime


class CleanupResult(BaseModel):
    """Result of a cleanup operation."""

    operation: str
    files_affected: int
    bytes_freed: int
    message: str


class HealthReport(BaseModel):
    """Complete health check report."""

    timestamp: datetime
    overall_status: ServiceStatus
    service_checks: list[ServiceCheckResult]
    cleanup_results: list[CleanupResult]
    total_duration_ms: float


class HealthChecker:
    """Runs API health checks and state cleanup."""

    def __init__(
        self,
        config: DaemonConfig,
        state: DaemonState,
        container: AppContainer | None = None,
        notification_service: NotificationService | None = None,
    ) -> None:
        """Initialize health checker.

        Args:
            config: Daemon configuration
            state: Current daemon state
            container: Optional DI container (auto-created if not provided)
            notification_service: Optional notification service for health alerts
        """
        from src.di.container import create_container

        self.config = config
        self.state = state
        self._container = container or create_container()
        self.notification_service = notification_service
        self._health_dir = Path(config.health.health_dir).expanduser()
        self._archive_dir = Path(config.health.archive_dir).expanduser()

    async def run(self) -> HealthReport:
        """Run all health checks and cleanup operations.

        Returns:
            HealthReport with results
        """
        start = time.perf_counter()

        # Run health checks sequentially to avoid rate limit burn
        checks = [
            await self._check_alpha_vantage(),
            await self._check_marketaux(),
            await self._check_alpaca(),
            await self._check_llm(),
            await self._check_finnhub(),
        ]

        # Run cleanup operations
        cleanups = [
            self._archive_old_analyses(),
            self._prune_stale_cache(),
            self._rotate_logs(),
            self._verify_state_integrity(),
        ]

        # Derive overall status (ignore SKIPPED)
        active_checks = [c for c in checks if c.status != ServiceStatus.SKIPPED]
        if any(c.status == ServiceStatus.UNHEALTHY for c in active_checks):
            overall = ServiceStatus.UNHEALTHY
        elif any(c.status == ServiceStatus.DEGRADED for c in active_checks):
            overall = ServiceStatus.DEGRADED
        elif active_checks:
            overall = ServiceStatus.HEALTHY
        else:
            overall = ServiceStatus.HEALTHY

        total_ms = (time.perf_counter() - start) * 1000

        report = HealthReport(
            timestamp=datetime.now(UTC),
            overall_status=overall,
            service_checks=checks,
            cleanup_results=cleanups,
            total_duration_ms=total_ms,
        )

        try:
            self._persist_report(report)
        except Exception as e:
            logger.error(f"Failed to persist report: {e}")

        try:
            self._prune_old_reports()
        except Exception as e:
            logger.error(f"Failed to prune old reports: {e}")

        # Send notification if health failures detected
        failed_services = [c for c in checks if c.status == ServiceStatus.UNHEALTHY]
        if failed_services and self.notification_service:
            await self._notify_health_failures(failed_services)

        return report

    async def _check_alpha_vantage(self) -> ServiceCheckResult:
        """Check Alpha Vantage API connectivity."""
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.SKIPPED,
                message="ALPHA_VANTAGE_API_KEY not configured",
                duration_ms=0,
                checked_at=datetime.now(UTC),
            )

        start = time.perf_counter()
        try:
            from alpha_vantage.timeseries import TimeSeries

            ts = TimeSeries(key=api_key)
            await asyncio.to_thread(ts.get_daily, "SPY", outputsize="compact")
            duration = (time.perf_counter() - start) * 1000
            return ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.HEALTHY,
                message="API responding normally",
                duration_ms=duration,
                checked_at=datetime.now(UTC),
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration,
                checked_at=datetime.now(UTC),
            )

    async def _check_marketaux(self) -> ServiceCheckResult:
        """Check Marketaux API connectivity."""
        api_key = os.getenv("MARKETAUX_API_KEY")
        if not api_key:
            return ServiceCheckResult(
                service="marketaux",
                status=ServiceStatus.SKIPPED,
                message="MARKETAUX_API_KEY not configured",
                duration_ms=0,
                checked_at=datetime.now(UTC),
            )

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.marketaux.com/v1/news/all",
                    params={"limit": 1, "api_token": api_key},
                )
                response.raise_for_status()
            duration = (time.perf_counter() - start) * 1000
            return ServiceCheckResult(
                service="marketaux",
                status=ServiceStatus.HEALTHY,
                message="API responding normally",
                duration_ms=duration,
                checked_at=datetime.now(UTC),
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return ServiceCheckResult(
                service="marketaux",
                status=ServiceStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration,
                checked_at=datetime.now(UTC),
            )

    async def _check_alpaca(self) -> ServiceCheckResult:
        """Check Alpaca API connectivity."""
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            return ServiceCheckResult(
                service="alpaca",
                status=ServiceStatus.SKIPPED,
                message="ALPACA_API_KEY/ALPACA_SECRET_KEY not configured",
                duration_ms=0,
                checked_at=datetime.now(UTC),
            )

        start = time.perf_counter()
        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
            await asyncio.to_thread(client.get_account)
            duration = (time.perf_counter() - start) * 1000
            return ServiceCheckResult(
                service="alpaca",
                status=ServiceStatus.HEALTHY,
                message="API responding normally",
                duration_ms=duration,
                checked_at=datetime.now(UTC),
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return ServiceCheckResult(
                service="alpaca",
                status=ServiceStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration,
                checked_at=datetime.now(UTC),
            )

    async def _check_llm(self) -> ServiceCheckResult:
        """Check LLM provider connectivity."""
        provider = self.config.llm.provider

        # Check API keys from config (with env var fallback for sensitive data)
        anthropic_key = self.config.api_keys.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        openai_key = self.config.api_keys.openai_api_key or os.getenv("OPENAI_API_KEY")

        # Check API keys for non-Ollama providers
        if provider == "anthropic" and not anthropic_key:
            return ServiceCheckResult(
                service=f"llm_{provider}",
                status=ServiceStatus.SKIPPED,
                message="ANTHROPIC_API_KEY not configured",
                duration_ms=0,
                checked_at=datetime.now(UTC),
            )
        if provider == "openai" and not openai_key:
            return ServiceCheckResult(
                service=f"llm_{provider}",
                status=ServiceStatus.SKIPPED,
                message="OPENAI_API_KEY not configured",
                duration_ms=0,
                checked_at=datetime.now(UTC),
            )

        start = time.perf_counter()
        try:
            if provider == "ollama":
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{base_url}/api/tags")
                    response.raise_for_status()
            else:
                llm = self._container.llm_client()
                try:
                    # Use default temperature (0.7) - no explicit override for health check
                    await llm.acomplete("Reply with OK")
                finally:
                    await llm.close()

            duration = (time.perf_counter() - start) * 1000
            return ServiceCheckResult(
                service=f"llm_{provider}",
                status=ServiceStatus.HEALTHY,
                message=f"{provider} responding normally",
                duration_ms=duration,
                checked_at=datetime.now(UTC),
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return ServiceCheckResult(
                service=f"llm_{provider}",
                status=ServiceStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration,
                checked_at=datetime.now(UTC),
            )

    async def _check_finnhub(self) -> ServiceCheckResult:
        """Check Finnhub API connectivity."""
        api_key = self.config.api_keys.finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        if not api_key:
            return ServiceCheckResult(
                service="finnhub",
                status=ServiceStatus.SKIPPED,
                message="Finnhub API key not configured",
                duration_ms=0,
                checked_at=datetime.now(UTC),
            )

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://finnhub.io/api/v1/news-sentiment",
                    params={"symbol": "SPY", "token": api_key},
                )
                response.raise_for_status()
            duration = (time.perf_counter() - start) * 1000
            return ServiceCheckResult(
                service="finnhub",
                status=ServiceStatus.HEALTHY,
                message="API responding normally",
                duration_ms=duration,
                checked_at=datetime.now(UTC),
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return ServiceCheckResult(
                service="finnhub",
                status=ServiceStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration,
                checked_at=datetime.now(UTC),
            )

    def _archive_old_analyses(self) -> CleanupResult:
        """Archive analyses older than archive_days from state."""
        cutoff = datetime.now(UTC) - timedelta(days=self.config.health.archive_days)
        old = [a for a in self.state.analyses if a.timestamp < cutoff]

        if not old:
            return CleanupResult(
                operation="archive_analyses",
                files_affected=0,
                bytes_freed=0,
                message="No old analyses to archive",
            )

        self._archive_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        archive_path = self._archive_dir / f"analyses-{date_str}.jsonl"

        bytes_written = 0
        with archive_path.open("a") as f:
            for record in old:
                line = record.model_dump_json() + "\n"
                f.write(line)
                bytes_written += len(line.encode())

        self.state.analyses = [a for a in self.state.analyses if a.timestamp >= cutoff]

        return CleanupResult(
            operation="archive_analyses",
            files_affected=1,
            bytes_freed=bytes_written,
            message=f"Archived {len(old)} analyses to {archive_path.name}",
        )

    def _prune_stale_cache(self) -> CleanupResult:
        """Prune expired entries from diskcache directories."""
        cache_base = Path("data/cache")
        if not cache_base.exists():
            return CleanupResult(
                operation="prune_cache",
                files_affected=0,
                bytes_freed=0,
                message="No cache directory found",
            )

        total_expired = 0
        for cache_dir in cache_base.iterdir():
            if not cache_dir.is_dir():
                continue
            try:
                from diskcache import Cache

                cache = Cache(str(cache_dir))
                expired = cache.expire()
                total_expired += expired
                cache.close()
            except Exception as e:
                logger.warning(f"Failed to prune cache {cache_dir}: {e}")

        return CleanupResult(
            operation="prune_cache",
            files_affected=total_expired,
            bytes_freed=0,
            message=f"Expired {total_expired} cache entries",
        )

    def _rotate_logs(self) -> CleanupResult:
        """Rotate log files exceeding max size."""
        max_bytes = self.config.health.log_max_size_mb * 1024 * 1024
        targets = [
            Path("logs/risk_audit.jsonl"),
            Path("logs/trades.jsonl"),
            Path("logs/execution_metrics.jsonl"),
            Path("~/.ai-casino/worker.log").expanduser(),
            Path("~/.ai-casino/tui.log").expanduser(),
        ]

        rotated = 0
        bytes_freed = 0
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")

        for target in targets:
            if not target.exists():
                continue
            size = target.stat().st_size
            if size <= max_bytes:
                continue

            rotated_name = f"{target.stem}.{date_str}{target.suffix}"
            rotated_path = target.parent / rotated_name
            target.rename(rotated_path)
            bytes_freed += size
            rotated += 1
            logger.info(f"Rotated {target} → {rotated_path} ({size} bytes)")

        return CleanupResult(
            operation="rotate_logs",
            files_affected=rotated,
            bytes_freed=0,
            message=(
                f"Rotated {rotated} log files ({bytes_freed} bytes)"
                if rotated
                else "No logs exceeded size limit"
            ),
        )

    def _verify_state_integrity(self) -> CleanupResult:
        """Verify daemon state file integrity."""
        state_path = Path(self.config.state.state_file).expanduser()

        if not state_path.exists():
            return CleanupResult(
                operation="verify_state",
                files_affected=0,
                bytes_freed=0,
                message="No state file to verify",
            )

        try:
            with state_path.open() as f:
                data = json.load(f)
            DaemonState.model_validate(data)
            return CleanupResult(
                operation="verify_state",
                files_affected=0,
                bytes_freed=0,
                message="State file valid",
            )
        except Exception as e:
            timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
            backup_path = state_path.with_suffix(f".corrupt.{timestamp}")
            state_path.rename(backup_path)
            logger.error(f"Corrupt state backed up to {backup_path}: {e}")
            return CleanupResult(
                operation="verify_state",
                files_affected=1,
                bytes_freed=0,
                message=f"Corrupt state backed up to {backup_path.name}",
            )

    def _persist_report(self, report: HealthReport) -> None:
        """Save health report to disk."""
        self._health_dir.mkdir(parents=True, exist_ok=True)
        date_str = report.timestamp.strftime("%Y-%m-%d")
        report_path = self._health_dir / f"health-{date_str}.json"
        report_path.write_text(report.model_dump_json(indent=2))
        logger.info(f"Health report saved to {report_path}")

    def _prune_old_reports(self) -> None:
        """Remove health reports older than 30 days."""
        if not self._health_dir.exists():
            return

        cutoff = datetime.now(UTC) - timedelta(days=30)
        for report_file in self._health_dir.glob("health-*.json"):
            try:
                date_str = report_file.stem.removeprefix("health-")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
                if file_date < cutoff:
                    report_file.unlink()
                    logger.debug(f"Pruned old health report: {report_file.name}")
            except ValueError:
                continue

    async def _notify_health_failures(self, failed: list[ServiceCheckResult]) -> None:
        """Send health failure notification.

        Args:
            failed: List of failed service checks
        """
        from src.daemon.config import NotificationTrigger
        from src.daemon.notifications import NotificationMessage

        if not self.notification_service:
            return

        services = ", ".join([f.service for f in failed])

        message = NotificationMessage(
            trigger=NotificationTrigger.HEALTH_FAILURE,
            title="API Health Check Failed",
            body=f"Services down: {services}",
            metadata={
                "symbol": "SYSTEM",
                "failed_services": [f.service for f in failed],
                "error_messages": [f.message for f in failed],
            },
            timestamp=datetime.now(UTC),
        )

        await self.notification_service.notify(NotificationTrigger.HEALTH_FAILURE, message)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"HealthChecker(health_dir={self._health_dir})"
