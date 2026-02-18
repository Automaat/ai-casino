"""Options flow data fetcher using yfinance."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field


class OptionContract(BaseModel):
    """Single options contract data."""

    strike: float
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    in_the_money: bool
    expiry: date
    option_type: str = Field(description="call or put")

    def __repr__(self) -> str:
        """String representation."""
        return f"OptionContract({self.option_type} {self.strike} exp={self.expiry} vol={self.volume})"


class OptionsChainSnapshot(BaseModel):
    """Aggregated options chain snapshot for a symbol."""

    symbol: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_call_volume: int = 0
    total_put_volume: int = 0
    total_call_oi: int = 0
    total_put_oi: int = 0
    near_term_expiry: date | None = None
    calls: list[OptionContract] = Field(default_factory=list)
    puts: list[OptionContract] = Field(default_factory=list)

    @property
    def total_volume(self) -> int:
        """Total options volume (calls + puts)."""
        return self.total_call_volume + self.total_put_volume

    @property
    def is_empty(self) -> bool:
        """Check if snapshot has no data."""
        return self.total_call_volume == 0 and self.total_put_volume == 0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptionsChainSnapshot({self.symbol} calls={self.total_call_volume} puts={self.total_put_volume})"
        )


class OptionsFlowFetcher:
    """Fetch options chain data from yfinance."""

    def __init__(self, max_expirations: int = 2) -> None:
        """Initialize options flow fetcher.

        Args:
            max_expirations: Maximum number of nearest expirations to fetch
        """
        self._max_expirations = max_expirations

    def fetch_options_chain(self, symbol: str) -> OptionsChainSnapshot:
        """Fetch options chain for symbol (sync, call via to_thread).

        Gets nearest non-expired expirations and aggregates volume/OI.

        Args:
            symbol: Stock ticker

        Returns:
            OptionsChainSnapshot with aggregated data
        """
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                logger.warning(f"No options expirations for {symbol}")
                return OptionsChainSnapshot(symbol=symbol)

            today = datetime.now(UTC).date()
            valid_expiries = [exp for exp in expirations if date.fromisoformat(exp) > today]

            if not valid_expiries:
                logger.warning(f"No future expirations for {symbol}")
                return OptionsChainSnapshot(symbol=symbol)

            selected = valid_expiries[: self._max_expirations]

            all_calls: list[OptionContract] = []
            all_puts: list[OptionContract] = []
            total_call_vol, total_put_vol = 0, 0
            total_call_oi, total_put_oi = 0, 0

            for exp_str in selected:
                exp_date = date.fromisoformat(exp_str)
                chain = ticker.option_chain(exp_str)

                calls_df: pd.DataFrame = chain[0].fillna(0)
                puts_df: pd.DataFrame = chain[1].fillna(0)

                for _, row in calls_df.iterrows():
                    vol = int(row.get("volume", 0))
                    oi = int(row.get("openInterest", 0))
                    total_call_vol += vol
                    total_call_oi += oi
                    all_calls.append(
                        OptionContract(
                            strike=float(row["strike"]),
                            last_price=float(row.get("lastPrice", 0)),
                            bid=float(row.get("bid", 0)),
                            ask=float(row.get("ask", 0)),
                            volume=vol,
                            open_interest=oi,
                            implied_volatility=float(row.get("impliedVolatility", 0)),
                            in_the_money=bool(row.get("inTheMoney", False)),
                            expiry=exp_date,
                            option_type="call",
                        )
                    )

                for _, row in puts_df.iterrows():
                    vol = int(row.get("volume", 0))
                    oi = int(row.get("openInterest", 0))
                    total_put_vol += vol
                    total_put_oi += oi
                    all_puts.append(
                        OptionContract(
                            strike=float(row["strike"]),
                            last_price=float(row.get("lastPrice", 0)),
                            bid=float(row.get("bid", 0)),
                            ask=float(row.get("ask", 0)),
                            volume=vol,
                            open_interest=oi,
                            implied_volatility=float(row.get("impliedVolatility", 0)),
                            in_the_money=bool(row.get("inTheMoney", False)),
                            expiry=exp_date,
                            option_type="put",
                        )
                    )

            near_term = date.fromisoformat(selected[0])

            logger.debug(
                f"Options chain for {symbol}: "
                f"calls_vol={total_call_vol} puts_vol={total_put_vol} "
                f"expirations={len(selected)}"
            )

            return OptionsChainSnapshot(
                symbol=symbol,
                total_call_volume=total_call_vol,
                total_put_volume=total_put_vol,
                total_call_oi=total_call_oi,
                total_put_oi=total_put_oi,
                near_term_expiry=near_term,
                calls=all_calls,
                puts=all_puts,
            )

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch options for {symbol}: {e}")
            return OptionsChainSnapshot(symbol=symbol)

    def __repr__(self) -> str:
        """String representation."""
        return f"OptionsFlowFetcher(max_expirations={self._max_expirations})"
