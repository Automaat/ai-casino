"""Shared utilities for extracting stock ticker symbols from text."""

import re


def extract_tickers(text: str, excluded_words: frozenset[str]) -> set[str]:
    """Extract stock tickers from text using regex.

    Args:
        text: Text to extract tickers from
        excluded_words: Common words to exclude from ticker detection

    Returns:
        Set of ticker symbols
    """
    tickers = set()
    # Match $SYMBOL or standalone 2-5 letter uppercase words
    pattern = r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b"

    for match in re.finditer(pattern, text):
        ticker = match.group(1) or match.group(2)
        if ticker and ticker not in excluded_words:
            tickers.add(ticker)

    return tickers
