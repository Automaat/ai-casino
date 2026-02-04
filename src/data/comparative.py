"""Comparative data fetcher for sector/market benchmarking using yfinance."""

from datetime import UTC, datetime, timedelta
from enum import StrEnum

import yfinance as yf
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(ValueError),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)


class Sector(StrEnum):
    """Sector to ETF mapping."""

    TECHNOLOGY = "XLK"
    HEALTHCARE = "XLV"
    FINANCIALS = "XLF"
    CONSUMER_DISCRETIONARY = "XLY"
    CONSUMER_STAPLES = "XLP"
    ENERGY = "XLE"
    INDUSTRIALS = "XLI"
    MATERIALS = "XLB"
    UTILITIES = "XLU"
    REAL_ESTATE = "XLRE"
    COMMUNICATION_SERVICES = "XLC"
    UNKNOWN = "SPY"  # Fallback to market index


# yfinance sector name to Sector enum mapping
SECTOR_MAPPING: dict[str, Sector] = {
    "technology": Sector.TECHNOLOGY,
    "healthcare": Sector.HEALTHCARE,
    "financial services": Sector.FINANCIALS,
    "financials": Sector.FINANCIALS,
    "consumer cyclical": Sector.CONSUMER_DISCRETIONARY,
    "consumer discretionary": Sector.CONSUMER_DISCRETIONARY,
    "consumer defensive": Sector.CONSUMER_STAPLES,
    "consumer staples": Sector.CONSUMER_STAPLES,
    "energy": Sector.ENERGY,
    "industrials": Sector.INDUSTRIALS,
    "basic materials": Sector.MATERIALS,
    "materials": Sector.MATERIALS,
    "utilities": Sector.UTILITIES,
    "real estate": Sector.REAL_ESTATE,
    "communication services": Sector.COMMUNICATION_SERVICES,
}

MARKET_INDEX = "SPY"


class StockInfo(BaseModel):
    """Basic stock information."""

    symbol: str
    sector: str | None
    industry: str | None
    pe_ratio: float | None
    price_to_book: float | None


class PerformanceData(BaseModel):
    """Performance metrics."""

    ytd_return: float | None
    three_month_return: float | None


class ComparativeData(BaseModel):
    """Comparative data for a stock vs sector and market."""

    stock_info: StockInfo
    stock_performance: PerformanceData
    sector_etf: str
    sector_pe: float | None
    sector_performance: PerformanceData
    market_pe: float | None
    market_performance: PerformanceData
    fetched_at: datetime

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class ComparativeDataFetcher:
    """Fetches comparative data using yfinance only (preserves Alpha Vantage quota)."""

    def __init__(self) -> None:
        """Initialize comparative data fetcher."""
        logger.info("Initialized ComparativeDataFetcher (yfinance only)")

    def fetch_comparative_data(self, symbol: str) -> ComparativeData:
        """Fetch stock data with sector and market comparisons.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ComparativeData with stock vs sector vs market metrics
        """
        logger.info(f"Fetching comparative data for {symbol}")

        stock_info = self._fetch_stock_info(symbol)
        stock_performance = self._fetch_performance(symbol)

        sector_etf = self._get_sector_etf(stock_info.sector)
        sector_pe = self._fetch_pe_ratio(sector_etf)
        sector_performance = self._fetch_performance(sector_etf)

        market_pe = self._fetch_pe_ratio(MARKET_INDEX)
        market_performance = self._fetch_performance(MARKET_INDEX)

        logger.info(
            f"Comparative data for {symbol}: sector={sector_etf}, "
            f"stock_pe={stock_info.pe_ratio}, sector_pe={sector_pe}, market_pe={market_pe}"
        )

        return ComparativeData(
            stock_info=stock_info,
            stock_performance=stock_performance,
            sector_etf=sector_etf,
            sector_pe=sector_pe,
            sector_performance=sector_performance,
            market_pe=market_pe,
            market_performance=market_performance,
            fetched_at=datetime.now(UTC),
        )

    @HTTP_RETRY
    def _fetch_stock_info(self, symbol: str) -> StockInfo:
        """Fetch basic stock information from yfinance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            StockInfo with sector, industry, P/E, P/B
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or "symbol" not in info:
            msg = f"No data available for {symbol}"
            raise ValueError(msg)

        return StockInfo(
            symbol=symbol,
            sector=info.get("sector"),
            industry=info.get("industry"),
            pe_ratio=self._safe_float(info.get("trailingPE") or info.get("forwardPE")),
            price_to_book=self._safe_float(info.get("priceToBook")),
        )

    @HTTP_RETRY
    def _fetch_performance(self, symbol: str) -> PerformanceData:
        """Fetch YTD and 3-month performance from price history.

        Args:
            symbol: Ticker symbol

        Returns:
            PerformanceData with returns
        """
        ticker = yf.Ticker(symbol)
        end_date = datetime.now(tz=UTC)
        ytd_start = datetime(end_date.year, 1, 1, tzinfo=UTC)
        three_month_start = end_date - timedelta(days=90)

        # Fetch YTD data
        ytd_data = ticker.history(start=ytd_start, end=end_date)
        ytd_return = None
        if len(ytd_data) >= 2:
            ytd_return = (ytd_data["Close"].iloc[-1] / ytd_data["Close"].iloc[0] - 1) * 100

        # Fetch 3-month data
        three_month_data = ticker.history(start=three_month_start, end=end_date)
        three_month_return = None
        if len(three_month_data) >= 2:
            three_month_return = (
                three_month_data["Close"].iloc[-1] / three_month_data["Close"].iloc[0] - 1
            ) * 100

        return PerformanceData(
            ytd_return=ytd_return,
            three_month_return=three_month_return,
        )

    @HTTP_RETRY
    def _fetch_pe_ratio(self, symbol: str) -> float | None:
        """Fetch P/E ratio for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            P/E ratio or None
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return self._safe_float(info.get("trailingPE") or info.get("forwardPE"))

    def _get_sector_etf(self, sector: str | None) -> str:
        """Map sector name to sector ETF ticker.

        Args:
            sector: Sector name from yfinance

        Returns:
            Sector ETF ticker (falls back to SPY if unknown)
        """
        if not sector:
            return Sector.UNKNOWN.value

        sector_lower = sector.lower()
        sector_enum = SECTOR_MAPPING.get(sector_lower, Sector.UNKNOWN)
        return sector_enum.value

    def _safe_float(self, value: float | str | None) -> float | None:
        """Safely convert value to float.

        Args:
            value: Value to convert

        Returns:
            Float or None if invalid
        """
        if value is None:
            return None
        try:
            result = float(value)
            # Filter out invalid P/E (negative or extreme values)
            if result <= 0 or result > 1000:
                return None
            return result
        except (ValueError, TypeError):
            return None

    def __repr__(self) -> str:
        """String representation."""
        return "ComparativeDataFetcher(source=yfinance)"
