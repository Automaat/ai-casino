"""Benchmark FinBERT local vs remote performance.

Compares inference latency between in-process model and microservice HTTP client.
"""

import asyncio
import os
import time
from statistics import mean, stdev


async def benchmark_local():
    """Benchmark local in-process FinBERT."""
    from src.models.sentiment import FinBERTSentiment, get_finbert_sentiment

    finbert_obj = get_finbert_sentiment(device="cpu")
    assert isinstance(finbert_obj, FinBERTSentiment)
    finbert = finbert_obj
    texts = [f"Sample financial text with earnings growth {i}" for i in range(50)]

    times = []
    for _ in range(10):
        start = time.perf_counter()
        finbert.analyze_batch(texts)
        times.append(time.perf_counter() - start)

    return {
        "mode": "local",
        "mean_ms": mean(times) * 1000,
        "stdev_ms": stdev(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


async def benchmark_remote():
    """Benchmark remote FinBERT microservice."""
    os.environ["FINBERT_MODE"] = "remote"
    os.environ["FINBERT_SERVICE_URL"] = "http://localhost:8485"

    from src.models.sentiment_client import FinBERTClient

    client = FinBERTClient("http://localhost:8485")
    texts = [f"Sample financial text with earnings growth {i}" for i in range(50)]

    times = []
    for _ in range(10):
        start = time.perf_counter()
        await client.analyze_batch_async(texts)
        times.append(time.perf_counter() - start)

    client.close()
    return {
        "mode": "remote",
        "mean_ms": mean(times) * 1000,
        "stdev_ms": stdev(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


async def main():
    """Run benchmarks and print comparison."""
    print("=" * 60)
    print("FinBERT Performance Benchmark")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  - Batch size: 50 texts")
    print("  - Iterations: 10")
    print("  - Device: CPU")
    print()

    print("Running local benchmark...")
    local = await benchmark_local()

    print("Running remote benchmark...")
    remote = await benchmark_remote()

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()
    print(f"Local (in-process):")
    print(f"  Mean:   {local['mean_ms']:.1f} ms ± {local['stdev_ms']:.1f} ms")
    print(f"  Min:    {local['min_ms']:.1f} ms")
    print(f"  Max:    {local['max_ms']:.1f} ms")
    print()
    print(f"Remote (microservice):")
    print(f"  Mean:   {remote['mean_ms']:.1f} ms ± {remote['stdev_ms']:.1f} ms")
    print(f"  Min:    {remote['min_ms']:.1f} ms")
    print(f"  Max:    {remote['max_ms']:.1f} ms")
    print()
    print("=" * 60)
    print("Analysis")
    print("=" * 60)
    overhead_ms = remote["mean_ms"] - local["mean_ms"]
    overhead_pct = (overhead_ms / local["mean_ms"]) * 100
    print(f"Overhead:        {overhead_ms:.1f} ms ({overhead_pct:.1f}%)")
    print(f"Target overhead: <25% (acceptable)")
    print()
    if overhead_pct < 25:
        print("✓ Remote overhead is acceptable")
    else:
        print("✗ Remote overhead exceeds target")
    print()


if __name__ == "__main__":
    asyncio.run(main())
