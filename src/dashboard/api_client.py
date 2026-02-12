"""Sync httpx client for daemon API."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.daemon.api import (
    AnalysesResponse,
    ConfigResponse,
    CorrelationMatrixResponse,
    DegradationHistoryResponse,
    DegradationResponse,
    EventResponse,
    ExecutionMetricsListResponse,
    FullConfigResponse,
    GamePlanResponse,
    HealthResponse,
    MarketEventsResponse,
    PositionsResponse,
    RebalanceResponse,
    RiskHistoryResponse,
    RiskReportResponse,
    SectorRotationResponse,
    SnapshotsResponse,
    StateSummaryResponse,
    WatchlistResponse,
)

_HTTP_SERVER_ERROR_MIN = 500


def _is_server_error(exception: BaseException) -> bool:
    """Check if exception is 5xx HTTP error."""
    return (
        isinstance(exception, httpx.HTTPStatusError)
        and exception.response.status_code >= _HTTP_SERVER_ERROR_MIN
    )


HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=(
        retry_if_exception_type(httpx.ConnectError)
        | retry_if_exception_type(httpx.TimeoutException)
        | retry_if_exception_type(httpx.ReadTimeout)
        | retry_if_exception_type(httpx.ReadError)
        | retry_if_exception(_is_server_error)
    ),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)


class DaemonAPIClient:
    """Sync httpx wrapper for daemon API."""

    def __init__(self, api_url: str) -> None:
        """Initialize API client.

        Args:
            api_url: Daemon API base URL
        """
        self.api_url = api_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)
        self._async_client = httpx.AsyncClient(timeout=10.0)
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._cache_ttl = timedelta(seconds=30)
        logger.info(f"Initialized DaemonAPIClient (api_url={self.api_url})")

    def _get_cached(self, key: str, fetch_fn: Callable[[], Any]) -> Any:  # noqa: ANN401
        """Generic cache wrapper.

        Args:
            key: Cache key
            fetch_fn: Function to fetch data if cache miss

        Returns:
            Cached or fresh data
        """
        now = datetime.now(UTC)
        if key in self._cache:
            ts, cached = self._cache[key]
            if now - ts < self._cache_ttl:
                logger.debug(f"Cache hit: {key}")
                return cached
        logger.debug(f"Cache miss: {key}")
        result = fetch_fn()
        self._cache[key] = (now, result)
        return result

    def is_healthy(self) -> bool:
        """Health check (returns False on exception, no raise).

        Returns:
            True if daemon is healthy
        """
        try:
            response = self._client.get(f"{self.api_url}/health")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    @HTTP_RETRY
    def get_health(self) -> HealthResponse:
        """Get daemon health status.

        Returns:
            HealthResponse
        """
        response = self._client.get(f"{self.api_url}/health")
        response.raise_for_status()
        return HealthResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_state_summary(self) -> StateSummaryResponse:
        """Get daemon state summary.

        Returns:
            StateSummaryResponse
        """
        response = self._client.get(f"{self.api_url}/state/summary")
        response.raise_for_status()
        return StateSummaryResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_config(self) -> ConfigResponse:
        """Get daemon configuration.

        Returns:
            ConfigResponse
        """
        response = self._client.get(f"{self.api_url}/config")
        response.raise_for_status()
        return ConfigResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_full_config(self) -> FullConfigResponse:
        """Get full daemon configuration with masked sensitive fields.

        Returns:
            FullConfigResponse
        """
        response = self._client.get(f"{self.api_url}/config/full")
        response.raise_for_status()
        return FullConfigResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_analyses(self, limit: int = 50, symbol: str | None = None) -> AnalysesResponse:
        """Get analysis history.

        Args:
            limit: Max number of analyses to return
            symbol: Filter by symbol (optional)

        Returns:
            AnalysesResponse
        """
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol  # type: ignore[assignment]

        response = self._client.get(f"{self.api_url}/analyses", params=params)
        response.raise_for_status()
        return AnalysesResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_positions(self) -> PositionsResponse:
        """Get active positions (cached 30s).

        Returns:
            PositionsResponse
        """
        return self._get_cached("positions", self._fetch_positions)

    def _fetch_positions(self) -> PositionsResponse:
        """Fetch positions from API.

        Returns:
            PositionsResponse
        """
        response = self._client.get(f"{self.api_url}/positions")
        response.raise_for_status()
        return PositionsResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_watchlist(self) -> WatchlistResponse:
        """Get merged watchlist.

        Returns:
            WatchlistResponse
        """
        response = self._client.get(f"{self.api_url}/watchlist")
        response.raise_for_status()
        return WatchlistResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_risk(self) -> RiskReportResponse | None:
        """Get latest risk report.

        Returns:
            RiskReportResponse or None if no report available
        """
        response = self._client.get(f"{self.api_url}/risk")
        response.raise_for_status()
        data = response.json()
        return RiskReportResponse.model_validate(data) if data else None

    @HTTP_RETRY
    def get_risk_history(self) -> RiskHistoryResponse:
        """Get historical risk reports.

        Returns:
            RiskHistoryResponse with list of historical reports
        """
        response = self._client.get(f"{self.api_url}/risk/history")
        response.raise_for_status()
        return RiskHistoryResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_sector_rotation(self) -> SectorRotationResponse | None:
        """Get latest sector rotation analysis.

        Returns:
            SectorRotationResponse or None if no data available
        """
        response = self._client.get(f"{self.api_url}/sector-rotation/latest")
        response.raise_for_status()
        data = response.json()
        return SectorRotationResponse.model_validate(data) if data else None

    @HTTP_RETRY
    def get_correlation_matrix(self) -> CorrelationMatrixResponse | None:
        """Get latest correlation matrix (cached for 30s).

        Returns:
            CorrelationMatrixResponse or None if no data available
        """
        return self._get_cached("correlation", self._fetch_correlation)

    def _fetch_correlation(self) -> CorrelationMatrixResponse | None:
        """Fetch correlation matrix from API.

        Returns:
            CorrelationMatrixResponse or None if no data available
        """
        response = self._client.get(f"{self.api_url}/correlation/latest")
        response.raise_for_status()
        data = response.json()
        return CorrelationMatrixResponse.model_validate(data) if data else None

    @HTTP_RETRY
    def get_degradation(self) -> DegradationResponse:
        """Get degradation status.

        Returns:
            DegradationResponse
        """
        response = self._client.get(f"{self.api_url}/degradation")
        response.raise_for_status()
        return DegradationResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_events(self, limit: int = 100) -> EventResponse:
        """Get event history.

        Args:
            limit: Max number of events to return

        Returns:
            EventResponse
        """
        response = self._client.get(f"{self.api_url}/events", params={"limit": limit})
        response.raise_for_status()
        return EventResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_market_events(self, limit: int = 100) -> MarketEventsResponse:
        """Get market events.

        Args:
            limit: Max number of events to return

        Returns:
            MarketEventsResponse
        """
        response = self._client.get(f"{self.api_url}/events/market", params={"limit": limit})
        response.raise_for_status()
        return MarketEventsResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_degradation_history(self, limit: int = 50) -> DegradationHistoryResponse:
        """Get degradation history.

        Args:
            limit: Max number of records to return

        Returns:
            DegradationHistoryResponse
        """
        response = self._client.get(f"{self.api_url}/events/degradation-history", params={"limit": limit})
        response.raise_for_status()
        return DegradationHistoryResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_game_plan(self) -> GamePlanResponse | None:
        """Get latest game plan.

        Returns:
            GamePlanResponse or None if game plan not available
        """
        response = self._client.get(f"{self.api_url}/game-plan")
        response.raise_for_status()
        data = response.json()
        return GamePlanResponse.model_validate(data) if data else None

    @HTTP_RETRY
    def get_snapshots(self, days: int = 30) -> SnapshotsResponse:
        """Get portfolio snapshots history (cached for 30s).

        Args:
            days: Number of days to look back

        Returns:
            SnapshotsResponse
        """
        return self._get_cached(f"snapshots_{days}", lambda: self._fetch_snapshots(days))

    def _fetch_snapshots(self, days: int) -> SnapshotsResponse:
        """Fetch snapshots from API.

        Args:
            days: Number of days to look back

        Returns:
            SnapshotsResponse
        """
        response = self._client.get(f"{self.api_url}/portfolio/snapshots", params={"days": days})
        response.raise_for_status()
        return SnapshotsResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_rebalance(self) -> RebalanceResponse | None:
        """Get latest portfolio rebalance data (cached 30s).

        Returns:
            RebalanceResponse or None if no rebalancing data
        """
        return self._get_cached("rebalance", self._fetch_rebalance)

    def _fetch_rebalance(self) -> RebalanceResponse | None:
        """Fetch rebalance from API.

        Returns:
            RebalanceResponse or None if no rebalancing data
        """
        response = self._client.get(f"{self.api_url}/portfolio/rebalance")
        response.raise_for_status()
        data = response.json()
        return RebalanceResponse.model_validate(data) if data else None

    @HTTP_RETRY
    def get_execution_metrics(self, limit: int = 50) -> ExecutionMetricsListResponse:
        """Get recent execution metrics (cached for 30s).

        Args:
            limit: Max number of metrics to return

        Returns:
            ExecutionMetricsListResponse
        """
        return self._get_cached(f"execution_metrics_{limit}", lambda: self._fetch_execution_metrics(limit))

    def _fetch_execution_metrics(self, limit: int) -> ExecutionMetricsListResponse:
        """Fetch execution metrics from API.

        Args:
            limit: Max number of metrics to return

        Returns:
            ExecutionMetricsListResponse
        """
        response = self._client.get(f"{self.api_url}/api/execution-metrics", params={"limit": limit})
        response.raise_for_status()
        return ExecutionMetricsListResponse.model_validate(response.json())

    @HTTP_RETRY
    def get_execution_metric_detail(self, workflow_id: str) -> dict:
        """Get single workflow execution detail (no cache).

        Args:
            workflow_id: Workflow ID to fetch

        Returns:
            WorkflowExecutionMetrics as dict
        """
        response = self._client.get(f"{self.api_url}/api/execution-metrics/{workflow_id}")
        response.raise_for_status()
        return response.json()

    async def aget_portfolio_data_parallel(self, days: int = 30) -> dict[str, Any]:
        """Fetch positions, snapshots, rebalance in parallel.

        Args:
            days: Number of days for snapshots

        Returns:
            Dict with keys: positions, snapshots, rebalance
        """

        async def safe_get_positions() -> httpx.Response | Exception:
            try:
                return await self._async_client.get(f"{self.api_url}/positions")
            except Exception as e:
                return e

        async def safe_get_snapshots() -> httpx.Response | Exception:
            try:
                return await self._async_client.get(
                    f"{self.api_url}/portfolio/snapshots", params={"days": days}
                )
            except Exception as e:
                return e

        async def safe_get_rebalance() -> httpx.Response | Exception:
            try:
                return await self._async_client.get(f"{self.api_url}/portfolio/rebalance")
            except Exception as e:
                return e

        async with asyncio.TaskGroup() as tg:
            positions_task = tg.create_task(safe_get_positions())
            snapshots_task = tg.create_task(safe_get_snapshots())
            rebalance_task = tg.create_task(safe_get_rebalance())

        positions = self._parse_positions(positions_task.result())
        snapshots = self._parse_snapshots(snapshots_task.result())
        rebalance = self._parse_rebalance(rebalance_task.result())

        return {"positions": positions, "snapshots": snapshots, "rebalance": rebalance}

    def get_portfolio_data_parallel(self, days: int = 30) -> dict[str, Any]:
        """Sync wrapper for parallel portfolio data fetch.

        Args:
            days: Number of days for snapshots

        Returns:
            Dict with keys: positions, snapshots, rebalance
        """
        # Use sync methods to avoid event loop conflicts in Dash callbacks
        positions = None
        try:
            positions = self.get_positions()
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch positions: {e}")

        snapshots = None
        try:
            snapshots = self.get_snapshots(days)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch snapshots: {e}")

        rebalance = None
        try:
            rebalance = self.get_rebalance()
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch rebalance: {e}")

        return {"positions": positions, "snapshots": snapshots, "rebalance": rebalance}

    def close(self) -> None:
        """Close HTTP clients."""
        self._client.close()
        try:
            asyncio.run(self._async_client.aclose())
        except RuntimeError:
            # Already in async context, schedule aclose() on existing loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = loop.create_task(self._async_client.aclose())
                # Store reference to avoid RUF006 warning (fire-and-forget cleanup)
                _ = task
            else:
                loop.run_until_complete(self._async_client.aclose())

    def _parse_positions(self, response: httpx.Response | Exception) -> PositionsResponse | None:
        """Parse positions response, handle errors."""
        if isinstance(response, httpx.Response):
            try:
                response.raise_for_status()
                return PositionsResponse.model_validate(response.json())
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to parse positions: {e}")
        elif isinstance(response, Exception):
            logger.warning(f"Failed to fetch positions: {response}")
        return None

    def _parse_snapshots(self, response: httpx.Response | Exception) -> SnapshotsResponse | None:
        """Parse snapshots response, handle errors."""
        if isinstance(response, httpx.Response):
            try:
                response.raise_for_status()
                return SnapshotsResponse.model_validate(response.json())
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to parse snapshots: {e}")
        elif isinstance(response, Exception):
            logger.warning(f"Failed to fetch snapshots: {response}")
        return None

    def _parse_rebalance(self, response: httpx.Response | Exception) -> RebalanceResponse | None:
        """Parse rebalance response, handle errors."""
        if isinstance(response, httpx.Response):
            try:
                response.raise_for_status()
                data = response.json()
                return RebalanceResponse.model_validate(data) if data else None
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to parse rebalance: {e}")
        elif isinstance(response, Exception):
            logger.warning(f"Failed to fetch rebalance: {response}")
        return None

    def __repr__(self) -> str:
        """String representation."""
        return f"DaemonAPIClient(api_url={self.api_url})"
