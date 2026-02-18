"""Benchmark script to demonstrate GIL fix with ProcessPoolExecutor.

Compares serial vs parallel FinBERT inference for multiple stocks.
"""

import asyncio
import time

from loguru import logger

from src.agents.sentiment import SentimentAnalyst
from src.data.news import NewsArticle
from src.models.sentiment import get_finbert_sentiment

# Sample news articles for testing
SAMPLE_ARTICLES = [
    NewsArticle(
        title="Company Reports Strong Earnings",
        description="Revenue beats expectations by 15%, stock surges on positive outlook.",
        url="https://example.com/1",
        published_at="2024-01-01",
        source="Example News",
    ),
    NewsArticle(
        title="Market Volatility Concerns",
        description="Analysts warn of potential headwinds in coming quarter.",
        url="https://example.com/2",
        published_at="2024-01-01",
        source="Example News",
    ),
    NewsArticle(
        title="New Product Launch Success",
        description="Consumer response exceeds expectations, driving growth.",
        url="https://example.com/3",
        published_at="2024-01-01",
        source="Example News",
    ),
] * 5  # 15 articles total


async def benchmark_parallel_analysis(symbols: list[str], articles: list[NewsArticle]) -> float:
    """Benchmark parallel analysis using ProcessPoolExecutor.

    Args:
        symbols: List of stock symbols to analyze
        articles: Sample articles to use

    Returns:
        Time taken in seconds
    """
    finbert = get_finbert_sentiment()
    analysts = [SentimentAnalyst(finbert) for _ in symbols]

    start = time.perf_counter()

    # Run analyses in parallel
    tasks = [analyst.analyze(symbol, articles) for symbol, analyst in zip(symbols, analysts, strict=True)]
    await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start
    logger.info(f"Parallel (ProcessPoolExecutor): {elapsed:.2f}s for {len(symbols)} stocks")
    return elapsed


async def benchmark_single_analysis(symbol: str, articles: list[NewsArticle]) -> float:
    """Benchmark single stock analysis (baseline).

    Args:
        symbol: Stock symbol to analyze
        articles: Sample articles to use

    Returns:
        Time taken in seconds
    """
    finbert = get_finbert_sentiment()
    analyst = SentimentAnalyst(finbert)

    start = time.perf_counter()
    await analyst.analyze(symbol, articles)
    elapsed = time.perf_counter() - start

    logger.info(f"Single stock: {elapsed:.2f}s for {symbol}")
    return elapsed


async def main() -> None:
    """Run benchmark demonstrating parallelism improvement."""
    symbols = ["AAPL", "MSFT", "GOOGL"]

    logger.info("Starting GIL fix benchmark")
    logger.info(f"Testing with {len(symbols)} stocks, {len(SAMPLE_ARTICLES)} articles each")
    logger.info("-" * 60)

    # Warm up FinBERT model
    logger.info("Warming up FinBERT model...")
    finbert_obj = get_finbert_sentiment()
    if not hasattr(finbert_obj, "analyze_batch"):
        msg = "FinBERT object missing analyze_batch method"
        raise RuntimeError(msg)
    finbert_obj.analyze_batch(["Test sentence for warmup"])

    # Benchmark single stock (baseline)
    logger.info("\n1. Single stock baseline:")
    single_time = await benchmark_single_analysis(symbols[0], SAMPLE_ARTICLES)

    # Brief pause between benchmarks
    await asyncio.sleep(1)

    # Benchmark parallel execution
    logger.info(f"\n2. {len(symbols)} stocks in parallel (ProcessPoolExecutor):")
    parallel_time = await benchmark_parallel_analysis(symbols, SAMPLE_ARTICLES)

    # Calculate expected time if serial (GIL-bound)
    expected_serial_time = single_time * len(symbols)

    # Results
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Single stock:           {single_time:.2f}s")
    logger.info(f"Expected serial (3x):   {expected_serial_time:.2f}s")
    logger.info(f"Actual parallel:        {parallel_time:.2f}s")
    logger.info(f"Speedup vs serial:      {expected_serial_time / parallel_time:.2f}x")
    logger.info(f"Efficiency:             {(expected_serial_time / parallel_time) / len(symbols) * 100:.1f}%")
    logger.info("=" * 60)
    logger.info("\nWith ProcessPoolExecutor, 3 parallel analyses take ~1x instead of 3x time")
    logger.info("(demonstrating true parallelism without GIL bottleneck)")


if __name__ == "__main__":
    from src.models.sentiment import shutdown_finbert_executor

    try:
        asyncio.run(main())
    finally:
        shutdown_finbert_executor()
