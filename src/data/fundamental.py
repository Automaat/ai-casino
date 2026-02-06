"""Fundamental data fetcher using Alpha Vantage API."""

import os
from typing import Any

from alpha_vantage.fundamentaldata import FundamentalData
from loguru import logger

from src.metrics.execution import timed_operation


class FundamentalDataFetcher:
    """Fetches fundamental company data via Alpha Vantage."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the fundamental data fetcher.

        Args:
            api_key: Alpha Vantage API key (defaults to env var)
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            msg = "ALPHA_VANTAGE_API_KEY required for fundamental data"
            raise ValueError(msg)

        self.fd = FundamentalData(key=self.api_key, output_format="json")
        logger.info("Initialized FundamentalDataFetcher")

    def fetch_overview(self, symbol: str) -> dict[str, Any]:
        """Fetch company overview with fundamental metrics.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing 50+ fundamental data fields

        Raises:
            ValueError: If no data available for symbol
            Exception: On API errors
        """
        with timed_operation("fundamental_data_fetch", source="alpha_vantage"):
            try:
                logger.info(f"Fetching fundamental overview for {symbol}")
                data, _ = self.fd.get_company_overview(symbol)

                if not data or "Symbol" not in data:
                    msg = f"No fundamental data available for {symbol}"
                    raise ValueError(msg)

                logger.info(f"Retrieved {len(data)} fundamental fields for {symbol}")
                return data

            except Exception as e:
                logger.error(f"Failed to fetch fundamental data for {symbol}: {e}")
                raise

    def __repr__(self) -> str:
        """Return string representation."""
        return f"FundamentalDataFetcher(api_key={'***' if self.api_key else None})"
