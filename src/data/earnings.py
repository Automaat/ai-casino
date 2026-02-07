"""Earnings calendar data fetcher."""

import time
from datetime import date, datetime

import yfinance as yf
from loguru import logger
from pydantic import BaseModel

from src.data.market import HTTP_RETRY


class EarningsEvent(BaseModel):
    """Single earnings event for a symbol."""

    symbol: str
    earnings_date: date
    estimate_eps: float | None = None


class EarningsCalendar(BaseModel):
    """Collection of earnings events."""

    events: list[EarningsEvent]
    fetched_at: datetime


class EarningsCalendarFetcher:
    """Fetch earnings dates from yfinance."""

    def __init__(self, delay_seconds: float = 0.5) -> None:
        """Initialize earnings calendar fetcher.

        Args:
            delay_seconds: Delay between sequential API calls for rate limiting
        """
        self._delay = delay_seconds
        logger.info("Initialized EarningsCalendarFetcher")

    def fetch_earnings_dates(self, symbols: list[str]) -> EarningsCalendar:
        """Fetch next earnings dates for a list of symbols.

        Args:
            symbols: Stock ticker symbols to fetch earnings for

        Returns:
            EarningsCalendar with events for symbols that have upcoming earnings
        """
        events: list[EarningsEvent] = []

        for i, symbol in enumerate(symbols):
            if i > 0:
                time.sleep(self._delay)

            event = self._fetch_single(symbol)
            if event:
                events.append(event)

        logger.info(f"Fetched earnings dates: {len(events)}/{len(symbols)} symbols have upcoming earnings")
        return EarningsCalendar(events=events, fetched_at=datetime.now())

    @HTTP_RETRY
    def _fetch_single(self, symbol: str) -> EarningsEvent | None:
        """Fetch earnings date for a single symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            EarningsEvent if upcoming earnings found, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar

            if calendar is None or (hasattr(calendar, "empty") and calendar.empty):
                logger.debug(f"No earnings calendar for {symbol}")
                return None

            # yfinance returns calendar as dict or DataFrame depending on version
            earnings_date = self._extract_earnings_date(calendar)
            if earnings_date is None:
                return None

            estimate_eps = self._extract_eps_estimate(calendar)

            logger.debug(f"{symbol} earnings date: {earnings_date}")
            return EarningsEvent(
                symbol=symbol,
                earnings_date=earnings_date,
                estimate_eps=estimate_eps,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch earnings for {symbol}: {e}")
            return None

    def _extract_earnings_date(self, calendar: object) -> date | None:
        """Extract earnings date from yfinance calendar data.

        Args:
            calendar: Calendar data from yfinance (dict or DataFrame)

        Returns:
            Earnings date or None
        """
        if isinstance(calendar, dict):
            raw = calendar.get("Earnings Date")
            if raw is None:
                return None
            # Can be a list of dates or a single date
            if isinstance(raw, list) and raw:
                return self._parse_date(raw[0])
            return self._parse_date(raw)

        # DataFrame format
        try:
            if "Earnings Date" in calendar.index:
                raw = calendar.loc["Earnings Date"]
                if hasattr(raw, "iloc"):
                    return self._parse_date(raw.iloc[0])
                return self._parse_date(raw)
        except (KeyError, IndexError):
            pass

        return None

    def _extract_eps_estimate(self, calendar: object) -> float | None:
        """Extract EPS estimate from yfinance calendar data.

        Args:
            calendar: Calendar data from yfinance

        Returns:
            EPS estimate or None
        """
        if isinstance(calendar, dict):
            eps = calendar.get("Earnings Average") or calendar.get("EPS Estimate")
            if eps is not None:
                try:
                    return float(eps)
                except (TypeError, ValueError):
                    return None
            return None

        # DataFrame format
        for key in ("Earnings Average", "EPS Estimate"):
            try:
                if key in calendar.index:
                    val = calendar.loc[key]
                    if hasattr(val, "iloc"):
                        val = val.iloc[0]
                    return float(val)
            except (KeyError, IndexError, TypeError, ValueError):
                continue

        return None

    def _parse_date(self, raw: object) -> date | None:
        """Parse a date from various formats.

        Args:
            raw: Date value (datetime, Timestamp, str, etc.)

        Returns:
            date object or None
        """
        if raw is None:
            return None

        if isinstance(raw, datetime):
            return raw.date()

        if isinstance(raw, date):
            return raw

        # pandas Timestamp or other date-like objects
        if hasattr(raw, "date"):
            return raw.date()

        # String fallback: try ISO format parsing
        try:
            return date.fromisoformat(str(raw))
        except ValueError:
            logger.warning(f"Unparseable earnings date: {raw}")
            return None

    def __repr__(self) -> str:
        """String representation."""
        return f"EarningsCalendarFetcher(delay={self._delay}s)"
