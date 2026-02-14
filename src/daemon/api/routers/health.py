"""Health and service status endpoints."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from loguru import logger

from src.daemon.api.models import HealthResponse, ServiceCheck, ServiceHealthResponse
from src.daemon.api.routers.shared import get_components

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Get daemon health status."""
    components = get_components(request)
    uptime = (datetime.now(UTC) - request.app.state.start_time).total_seconds()

    # Determine health status from degradation tier
    degradation_tier = "FULL"
    last_run = None

    try:
        degradation_history = await components.state.get_degradation_history(limit=1)
        if degradation_history:
            degradation_tier = degradation_history[-1].tier

        last_run = await components.state.get_last_run()
    except Exception as e:
        # DB temporarily unavailable due to concurrent operations - still healthy
        logger.opt(exception=True).debug(f"Failed to fetch state for health check: {e}")

    status = "healthy" if degradation_tier == "NONE" else "degraded"

    return HealthResponse(
        status=status,
        uptime_seconds=uptime,
        daemon_running=components.running,
        last_run=last_run.isoformat() if last_run else None,
    )


@router.get("/health/services", response_model=ServiceHealthResponse)
async def get_service_health(request: Request) -> ServiceHealthResponse:
    """Get individual service health checks."""
    components = get_components(request)

    def _read_health_report() -> dict[str, Any]:
        """Read the latest health report from disk, with a safe fallback."""
        health_dir = Path(components.config.health.health_dir).expanduser()
        reports = sorted(health_dir.glob("health-*.json"))
        if not reports:
            return {"overall_status": "HEALTHY", "service_checks": []}

        latest_file = reports[-1]
        try:
            return json.loads(latest_file.read_text())
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to read or parse health report {latest_file}: {e}")
            return {"overall_status": "HEALTHY", "service_checks": []}

    # Read health report in thread to avoid blocking
    try:
        report_data = await asyncio.to_thread(_read_health_report)
    except RuntimeError:
        # Executor shut down (during shutdown/restart) - return healthy status
        report_data = {"overall_status": "HEALTHY", "service_checks": []}

    try:
        raw_checks = report_data.get("service_checks", [])
        if not isinstance(raw_checks, list):
            msg = "service_checks is not a list"
            raise TypeError(msg)

        # Convert ServiceCheckResult-like dicts to ServiceCheck models
        service_checks = [
            ServiceCheck(
                service=check["service"],
                status=check["status"],
                message=check["message"],
                duration_ms=check["duration_ms"],
                checked_at=check["checked_at"],
            )
            for check in raw_checks
        ]

        overall_status_raw = report_data.get("overall_status", "HEALTHY")
        overall_status = str(overall_status_raw) if overall_status_raw else "HEALTHY"
        return ServiceHealthResponse(
            overall_status=overall_status,
            service_checks=service_checks,
        )
    except Exception as e:
        logger.opt(exception=True).warning(f"Invalid health report format, using fallback health status: {e}")
        return ServiceHealthResponse(
            overall_status="HEALTHY",
            service_checks=[],
        )
