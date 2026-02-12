#!/usr/bin/env python3
"""Profile daemon cycle to identify performance bottlenecks.

This script runs a single daemon cycle with profiling enabled, analyzes the results,
and generates a report identifying bottlenecks in:
- LLM API wait time
- FinBERT inference time
- Market data fetching
- Database writes
- pandas-ta calculations

Usage:
    python scripts/profile_daemon.py [--stocks AAPL,TSLA,GOOGL,MSFT,NVDA] [--max-concurrent 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.daemon.config import DaemonConfig
from src.daemon.factory import DaemonFactory

console = Console()


def create_profiling_config(stocks: list[str], max_concurrent: int) -> DaemonConfig:
    """Create daemon config with profiling enabled.

    Args:
        stocks: List of stock symbols to analyze
        max_concurrent: Max concurrent analyses

    Returns:
        DaemonConfig with profiling enabled
    """
    config = DaemonConfig()
    config.watchlist = stocks
    config.analysis_orchestration.max_concurrent_analyses = max_concurrent
    config.profiling.enabled = True
    config.profiling.output_dir = "~/.ai-casino/profiles/benchmark"
    config.profiling.sample_rate = 1
    config.profiling.top_n_functions = 100
    config.logging.log_level = "INFO"

    # Disable optional components for cleaner profile
    config.screening.enabled = False
    config.rebalancing.enabled = False
    config.optimization.enabled = False
    config.discovery.enabled = False
    config.reporting.enabled = False
    config.game_plan.enabled = False

    return config


def get_profile_dir(config: DaemonConfig) -> Path:
    """Get expanded profile directory path.

    Args:
        config: Daemon configuration

    Returns:
        Expanded profile directory path
    """
    return Path(config.profiling.output_dir).expanduser()


async def run_profiled_cycle(config: DaemonConfig) -> tuple[float, Path]:
    """Run single daemon cycle with profiling.

    Args:
        config: Daemon configuration

    Returns:
        Tuple of (duration_seconds, profile_dir)
    """
    factory = DaemonFactory(config)
    components = factory.create_components()

    # Import here to avoid circular dependency
    from src.daemon.cycle_orchestrator import DaemonCycleOrchestrator

    cycle_orchestrator = DaemonCycleOrchestrator(
        components=components,
        task_runner=components.task_runner,
        runner=None,
        profiler=components.profiler,
    )

    console.print("[bold green]Starting profiled daemon cycle...[/bold green]")
    console.print(f"Analyzing: {', '.join(config.watchlist)}")
    console.print(f"Max concurrent: {config.analysis_orchestration.max_concurrent_analyses}")

    start_time = asyncio.get_event_loop().time()
    await cycle_orchestrator.run_cycle()
    duration = asyncio.get_event_loop().time() - start_time

    console.print(f"[green]✓[/green] Cycle complete in {duration:.2f}s")

    # Get profile directory
    profile_dir = get_profile_dir(config)
    return duration, profile_dir


def load_latest_profile(profile_dir: Path) -> dict | None:
    """Load latest JSON profile.

    Args:
        profile_dir: Profile directory

    Returns:
        Profile data or None if not found
    """
    json_files = sorted(profile_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        return None

    with json_files[0].open() as f:
        return json.load(f)


def analyze_bottlenecks(profile: dict) -> list[dict]:
    """Analyze profile and identify bottlenecks.

    Args:
        profile: Profile data

    Returns:
        List of bottleneck dicts sorted by cumtime
    """
    top_functions = profile.get("top_functions", [])

    # Categorize functions
    bottlenecks: list[dict[str, str | float | int]] = []
    for func in top_functions:
        func_name = str(func["function"])
        cumtime = float(func["cumtime"])
        ncalls = int(func["ncalls"])
        percall = float(func["percall"])

        # Categorize by pattern matching
        category = "other"
        if any(x in func_name for x in ["anthropic", "openai", "ollama", "llm", "complete"]):
            category = "llm_api"
        elif any(x in func_name for x in ["finbert", "sentiment", "transformers", "torch"]):
            category = "finbert"
        elif any(x in func_name for x in ["market", "yfinance", "alpha_vantage", "fetch"]):
            category = "market_data"
        elif any(x in func_name for x in ["database", "sqlite", "execute", "commit"]):
            category = "database"
        elif any(x in func_name for x in ["pandas_ta", "ta.", "rsi", "macd", "indicator"]):
            category = "technical_indicators"
        elif any(x in func_name for x in ["news", "marketaux", "article"]):
            category = "news_data"

        bottlenecks.append(
            {
                "function": func_name,
                "category": category,
                "cumtime": cumtime,
                "ncalls": ncalls,
                "percall": percall,
                "percent": 0.0,  # Filled later
            }
        )

    # Calculate percentages
    total_time = sum(float(b["cumtime"]) for b in bottlenecks[:50])  # Top 50 as baseline
    for bottleneck in bottlenecks:
        bottleneck["percent"] = (float(bottleneck["cumtime"]) / total_time * 100) if total_time > 0 else 0.0

    return sorted(bottlenecks, key=lambda x: float(x["cumtime"]), reverse=True)


def generate_report(duration: float, profile: dict, bottlenecks: list[dict], output_path: Path) -> None:
    """Generate bottleneck report.

    Args:
        duration: Total cycle duration
        profile: Profile data
        bottlenecks: Analyzed bottlenecks
        output_path: Output file path
    """
    # Console output
    console.print("\n[bold]Profiling Summary[/bold]")
    console.print(f"Total duration: {duration:.2f}s")
    console.print(f"Profiling overhead: {profile.get('profiling_overhead_percent', 0):.1f}%")
    console.print(f"Functions tracked: {len(profile.get('top_functions', []))}")

    # Top 10 bottlenecks table
    table = Table(title="\nTop 10 Bottlenecks")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Cumulative Time", justify="right", style="green")
    table.add_column("% of Total", justify="right", style="yellow")
    table.add_column("Calls", justify="right", style="blue")
    table.add_column("Per Call", justify="right", style="white")
    table.add_column("Function", style="dim", overflow="fold")

    for i, bottleneck in enumerate(bottlenecks[:10], 1):
        table.add_row(
            str(i),
            bottleneck["category"],
            f"{bottleneck['cumtime']:.3f}s",
            f"{bottleneck['percent']:.1f}%",
            str(bottleneck["ncalls"]),
            f"{bottleneck['percall']:.4f}s",
            bottleneck["function"],
        )

    console.print(table)

    # Category breakdown
    category_times = {}
    for bottleneck in bottlenecks[:50]:  # Top 50
        cat = bottleneck["category"]
        category_times[cat] = category_times.get(cat, 0) + bottleneck["cumtime"]

    cat_table = Table(title="\nTime by Category (Top 50 Functions)")
    cat_table.add_column("Category", style="magenta")
    cat_table.add_column("Total Time", justify="right", style="green")
    cat_table.add_column("% of Total", justify="right", style="yellow")

    total_cat_time = sum(category_times.values())
    for cat, time in sorted(category_times.items(), key=lambda x: x[1], reverse=True):
        percent = (time / total_cat_time * 100) if total_cat_time > 0 else 0.0
        cat_table.add_row(cat, f"{time:.3f}s", f"{percent:.1f}%")

    console.print(cat_table)

    # Save detailed report
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "summary": {
            "total_duration": duration,
            "profiling_overhead_percent": profile.get("profiling_overhead_percent", 0),
            "functions_tracked": len(profile.get("top_functions", [])),
        },
        "top_10_bottlenecks": bottlenecks[:10],
        "category_breakdown": {
            cat: {"time": time, "percent": (time / total_cat_time * 100) if total_cat_time > 0 else 0.0}
            for cat, time in category_times.items()
        },
        "all_bottlenecks": bottlenecks,
    }

    output_path.write_text(json.dumps(report, indent=2))
    console.print(f"\n[green]✓[/green] Detailed report saved to: {output_path}")


async def create_github_issues(bottlenecks: list[dict]) -> None:
    """Create GitHub issues for top 3 bottlenecks.

    Args:
        bottlenecks: Analyzed bottlenecks
    """
    console.print("\n[bold]Creating GitHub issues for top 3 bottlenecks...[/bold]")

    for i, bottleneck in enumerate(bottlenecks[:3], 1):
        category = bottleneck["category"]
        cumtime = bottleneck["cumtime"]
        percent = bottleneck["percent"]
        function = bottleneck["function"]

        # Create issue title
        title = f"perf(daemon): optimize {category.replace('_', ' ')} ({percent:.1f}% of cycle time)"

        # Create issue body
        body = f"""## Problem

Profiling identified {category.replace("_", " ")} as bottleneck #{i} in daemon cycle.

## Metrics

- **Cumulative time:** {cumtime:.3f}s
- **Percentage of cycle:** {percent:.1f}%
- **Top function:** `{function}`

## Tasks

1. Profile specific component in isolation
2. Identify optimization opportunities (caching, batching, async improvements)
3. Implement optimizations
4. Benchmark improvement (target: reduce by 30%+)

## Related

Part of profiling effort from #520
"""

        # Create issue via gh CLI
        import shutil
        import subprocess

        gh_path = shutil.which("gh")
        if not gh_path:
            console.print("[red]✗[/red] gh CLI not found in PATH")
            continue

        result = await asyncio.to_thread(
            subprocess.run,
            [gh_path, "issue", "create", "--title", title, "--body", body, "--label", "performance"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            issue_url = result.stdout.strip()
            console.print(f"[green]✓[/green] Created issue #{i}: {issue_url}")
        else:
            console.print(f"[red]✗[/red] Failed to create issue #{i}: {result.stderr}")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile daemon cycle for bottlenecks")
    parser.add_argument(
        "--stocks",
        default="AAPL,TSLA,GOOGL,MSFT,NVDA",
        help="Comma-separated list of stocks to analyze",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent analyses",
    )
    parser.add_argument(
        "--no-issues",
        action="store_true",
        help="Skip creating GitHub issues",
    )
    args = parser.parse_args()

    stocks = [s.strip() for s in args.stocks.split(",")]

    # Create config
    config = create_profiling_config(stocks, args.max_concurrent)

    # Run profiled cycle
    try:
        duration, profile_dir = await run_profiled_cycle(config)
    except Exception as e:
        console.print(f"[red]✗[/red] Profiling failed: {e}")
        logger.exception("Profiling failed")
        sys.exit(1)

    # Load and analyze results
    profile = load_latest_profile(profile_dir)
    if not profile:
        console.print(f"[red]✗[/red] No profile data found in {profile_dir}")
        sys.exit(1)

    bottlenecks = analyze_bottlenecks(profile)

    # Generate report
    report_path = profile_dir / f"bottleneck_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    generate_report(duration, profile, bottlenecks, report_path)

    # Create GitHub issues
    if not args.no_issues:
        await create_github_issues(bottlenecks)

    console.print("\n[bold green]✓ Profiling complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
