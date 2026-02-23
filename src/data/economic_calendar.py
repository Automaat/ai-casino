"""FRED economic calendar data fetcher."""

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Final

import httpx
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.cache.memory import MemoryTTLCache

HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(ValueError),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)

FRED_RELEASES: Final = {
    10: ("CPI", "high"),  # Consumer Price Index
    50: ("Nonfarm Payroll", "high"),  # Employment Situation
    17: ("GDP", "medium"),  # Gross Domestic Product
}

# FRED releases are typically published at 08:30 ET = 13:30 UTC
FRED_RELEASE_HOUR_UTC: Final = 13
FRED_RELEASE_MINUTE_UTC: Final = 30

# Hardcoded 2026 FOMC decision dates (published annually by Federal Reserve)
# Rate decisions announced at 14:00 UTC
FOMC_DATES_2026: Final = [
    datetime(2026, 1, 28, 14, 0, tzinfo=UTC),
    datetime(2026, 3, 18, 14, 0, tzinfo=UTC),
    datetime(2026, 4, 29, 14, 0, tzinfo=UTC),
    datetime(2026, 6, 10, 14, 0, tzinfo=UTC),
    datetime(2026, 7, 29, 14, 0, tzinfo=UTC),
    datetime(2026, 9, 16, 14, 0, tzinfo=UTC),
    datetime(2026, 10, 28, 14, 0, tzinfo=UTC),
    datetime(2026, 12, 9, 14, 0, tzinfo=UTC),
]


class EconomicCalendarEntry(BaseModel):
    """Single economic calendar entry."""

    country: str = "US"
    event: str
    impact: str  # "high" | "medium"
    scheduled_at: datetime
    actual: str | None = None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EconomicCalendarEntry(event={self.event}, impact={self.impact}, at={self.scheduled_at.date()})"
        )


class EconomicCalendarFetcher:
    """Fetch economic calendar data from FRED API and hardcoded FOMC dates."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(
        self,
        api_key: str | None,
        cache_ttl: int = 1800,
    ) -> None:
        """Initialize FRED economic calendar fetcher.

        Args:
            api_key: FRED API key (free from fred.stlouisfed.org)
            cache_ttl: Cache TTL in seconds
        """
        self._api_key = api_key
        self._cache_ttl = cache_ttl
        self._cache = MemoryTTLCache()

        if not self._api_key:
            logger.warning("fred_api_key not configured - only FOMC hardcoded dates available")
        else:
            logger.info("Initialized EconomicCalendarFetcher")

    def _cache_key(self, *args: str) -> str:
        """Generate cache key from args."""
        raw = ":".join(str(a) for a in args)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @HTTP_RETRY
    def _fetch_release_dates(self, release_id: int, from_date: str, to_date: str) -> list[str]:
        """Fetch release dates from FRED API for a given release ID.

        Args:
            release_id: FRED release ID
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            List of date strings (YYYY-MM-DD), empty if api_key not set
        """
        if not self._api_key:
            return []

        params = {
            "release_id": release_id,
            "realtime_start": from_date,
            "realtime_end": to_date,
            "include_release_dates_with_no_data": "true",
            "api_key": self._api_key,
            "file_type": "json",
        }

        url = f"{self.BASE_URL}/release/dates"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            return [item["date"] for item in data.get("release_dates", [])]

        except httpx.HTTPStatusError as e:
            logger.opt(exception=True).error(
                f"FRED API error for release {release_id}: HTTP {e.response.status_code}\n"
                f"Response: {e.response.text[:300]}"
            )
            raise
        except Exception as e:
            logger.opt(exception=True).error(f"FRED fetch failed for release {release_id}: {e}")
            raise

    def fetch_economic_calendar(self, from_date: str, to_date: str) -> list[EconomicCalendarEntry]:
        """Fetch upcoming economic events combining FRED releases and FOMC dates.

        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            List of EconomicCalendarEntry sorted by scheduled_at
        """
        cache_key = self._cache_key("fred_calendar", from_date, to_date)
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for economic calendar {from_date} → {to_date}")
            return [EconomicCalendarEntry.model_validate(e) for e in cached]

        entries: list[EconomicCalendarEntry] = []

        # Fetch FRED releases
        for release_id, (event_name, impact) in FRED_RELEASES.items():
            try:
                dates = self._fetch_release_dates(release_id, from_date, to_date)
                for date_str in dates:
                    parts = date_str.split("-")
                    scheduled = datetime(
                        int(parts[0]),
                        int(parts[1]),
                        int(parts[2]),
                        FRED_RELEASE_HOUR_UTC,
                        FRED_RELEASE_MINUTE_UTC,
                        tzinfo=UTC,
                    )
                    entries.append(
                        EconomicCalendarEntry(
                            event=event_name,
                            impact=impact,
                            scheduled_at=scheduled,
                        )
                    )
            except Exception as e:
                logger.opt(exception=True).warning(f"Skipping FRED release {release_id} ({event_name}): {e}")

        # Add hardcoded FOMC dates within window
        from_dt = datetime.fromisoformat(from_date).replace(tzinfo=UTC)
        to_dt = datetime.fromisoformat(to_date).replace(tzinfo=UTC) + timedelta(days=1)

        if datetime.now(UTC).year > 2026:
            logger.warning(
                "FOMC_DATES_2026 is stale - current year exceeds 2026. "
                "Update FOMC_DATES_2026 in economic_calendar.py with dates from "
                "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
            )

        for fomc_dt in FOMC_DATES_2026:
            if from_dt <= fomc_dt <= to_dt:
                entries.append(
                    EconomicCalendarEntry(
                        event="FOMC Meeting",
                        impact="high",
                        scheduled_at=fomc_dt,
                    )
                )

        entries.sort(key=lambda e: e.scheduled_at)

        self._cache.set(cache_key, [e.model_dump(mode="json") for e in entries], expire=self._cache_ttl)
        logger.info(f"Fetched {len(entries)} economic events ({from_date} → {to_date})")
        return entries

    def __repr__(self) -> str:
        """String representation."""
        return f"EconomicCalendarFetcher(authenticated={bool(self._api_key)})"
