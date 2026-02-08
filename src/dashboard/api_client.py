"""Sync httpx client for daemon API."""

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
    DegradationResponse,
    EventResponse,
    GamePlanResponse,
    HealthResponse,
    PositionsResponse,
    RiskReportResponse,
    StateSummaryResponse,
    WatchlistResponse,
)

_HTTP_SERVER_ERROR_MIN = 500


def _is_server_error(exception: Exception) -> bool:
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
        logger.info(f"Initialized DaemonAPIClient (api_url={self.api_url})")

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
            params["symbol"] = symbol

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
    def get_game_plan(self) -> GamePlanResponse | None:
        """Get latest game plan.

        Returns:
            GamePlanResponse or None if game plan not available
        """
        response = self._client.get(f"{self.api_url}/game-plan")
        response.raise_for_status()
        data = response.json()
        return GamePlanResponse.model_validate(data) if data else None

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"DaemonAPIClient(api_url={self.api_url})"
