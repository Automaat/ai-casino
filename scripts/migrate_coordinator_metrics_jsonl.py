#!/usr/bin/env python3
"""Migrate coordinator-metrics.jsonl to PostgreSQL."""

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.di.container import create_container
from src.v1.coordinator.metrics import CoordinatorCycleMetrics


async def migrate_coordinator_metrics() -> tuple[int, int, int]:
    """Migrate JSONL to postgres.

    Returns:
        Tuple of (migrated count, skipped count, failed count)
    """
    jsonl_path = Path.home() / ".ai-casino" / "coordinator-metrics.jsonl"
    if not jsonl_path.exists():
        logger.warning(f"No JSONL file at {jsonl_path}")
        return (0, 0, 0)

    container = create_container()
    repo = container.coordinator_metrics_repository()

    migrated = 0
    skipped = 0
    failed = 0

    seen = set()

    with jsonl_path.open() as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                timestamp = datetime.fromisoformat(data["timestamp"])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)

                dedup_key = (
                    data["cycle_num"],
                    timestamp.isoformat(),
                )

                if dedup_key in seen:
                    skipped += 1
                    continue

                seen.add(dedup_key)

                metrics = CoordinatorCycleMetrics(
                    cycle_num=data["cycle_num"],
                    timestamp=timestamp,
                    symbols_analyzed=data.get("symbols_analyzed", []),
                    tool_calls_made=data["tool_calls_made"],
                    trades_proposed=data["trades_proposed"],
                    trades_executed=data["trades_executed"],
                    trades_pending=data.get("trades_pending", 0),
                    game_plan_generated=data["game_plan_generated"],
                    cycle_duration_seconds=data["cycle_duration_seconds"],
                    patterns_detected=data.get("patterns_detected", 0),
                )

                await repo.create(metrics)
                migrated += 1

                if migrated % 50 == 0:
                    logger.info(f"Migrated {migrated} records...")

            except Exception as e:
                logger.opt(exception=True).error(f"Line {line_num} failed: {e}")
                failed += 1

    if migrated > 0:
        backup_path = jsonl_path.with_suffix(".jsonl.bak")
        jsonl_path.rename(backup_path)
        logger.info(f"Backed up to {backup_path}")

    return (migrated, skipped, failed)


if __name__ == "__main__":
    migrated, skipped, failed = asyncio.run(migrate_coordinator_metrics())
    logger.info(f"Migration complete: {migrated} migrated, {skipped} skipped, {failed} failed")
