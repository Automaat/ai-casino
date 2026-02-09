"""Sync httpx client for daemon API."""

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
        """Get active positions.

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
        """Get latest portfolio rebalance data.

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

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"DaemonAPIClient(api_url={self.api_url})"
