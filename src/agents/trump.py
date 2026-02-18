"""Trump social media analysis models."""

from pydantic import BaseModel

from src.strategies.signal import Signal

# Market-relevant keywords that historically move markets
MARKET_KEYWORDS = frozenset(
    {
        # Direct trading signals
        "buy",
        "sell",
        "great time to buy",
        "invest",
        "stock",
        "market",
        "stocks",
        # Tariff/trade
        "tariff",
        "tariffs",
        "trade deal",
        "trade war",
        "china",
        "pause",
        "negotiations",
        # Economic policy
        "interest rates",
        "fed",
        "federal reserve",
        "inflation",
        "economy",
        "economic",
        "recession",
        "jobs",
        "unemployment",
        # Crypto
        "bitcoin",
        "btc",
        "crypto",
        "cryptocurrency",
        # Companies
        "tesla",
        "apple",
        "amazon",
        "google",
        "microsoft",
        "meta",
        "nvidia",
    }
)

# Company name to ticker mapping
COMPANY_TICKERS = {
    "tesla": "TSLA",
    "apple": "AAPL",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "microsoft": "MSFT",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "boeing": "BA",
    "ford": "F",
    "general motors": "GM",
    "disney": "DIS",
    "coinbase": "COIN",
    "truth social": "DJT",
    "trump media": "DJT",
}


class TrumpAnalysis(BaseModel):
    """Trump post analysis result."""

    market_relevant: bool
    signal: Signal
    mentioned_tickers: list[str]
    sentiment: str
    confidence: float
    key_phrases: list[str]
    interpretation: str
    post_count: int
