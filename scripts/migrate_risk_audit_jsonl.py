#!/usr/bin/env python3
"""Migrate logs/risk_audit.jsonl to PostgreSQL."""

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.agents.risk.models import RiskAuditRecord
from src.di.container import create_container
from src.strategies.signal import Signal


def _read_jsonl_lines(jsonl_path: Path) -> list[str] | None:
    """Read lines from JSONL file, returning None if file missing."""
    if not jsonl_path.exists():
        return None
    with jsonl_path.open() as f:
        return f.readlines()


def _backup_file(jsonl_path: Path) -> Path:
    """Rename JSONL file to .bak and return backup path."""
    backup_path = jsonl_path.with_suffix(".jsonl.bak")
    jsonl_path.rename(backup_path)
    return backup_path


async def migrate_risk_audit() -> tuple[int, int, int]:
    """Migrate JSONL to postgres.

    Returns:
        Tuple of (migrated count, skipped count, failed count)
    """
    jsonl_path = Path("logs/risk_audit.jsonl")
    lines = await asyncio.to_thread(_read_jsonl_lines, jsonl_path)
    if lines is None:
        logger.warning(f"No JSONL file at {jsonl_path}")
        return (0, 0, 0)

    container = create_container()
    repo = container.risk_audit_repository()

    migrated = 0
    skipped = 0
    failed = 0

    seen: set[tuple[str, str, str]] = set()

    for line_num, line in enumerate(lines, 1):
        try:
            data = json.loads(line.strip())

            timestamp = datetime.fromisoformat(data["timestamp"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)

            dedup_key = (
                timestamp.isoformat(),
                data["symbol"],
                data["action"],
            )

            if dedup_key in seen:
                skipped += 1
                continue

            seen.add(dedup_key)

            record = RiskAuditRecord(
                timestamp=timestamp,
                symbol=data["symbol"],
                action=Signal(data["action"]),
                current_price=data["current_price"],
                approved=data["approved"],
                risk_level=data["risk_level"],
                risk_score=data["risk_score"],
                confidence=data["confidence"],
                recommended_shares=data["recommended_shares"],
                position_value=data["position_value"],
                risk_amount=data["risk_amount"],
                risk_percent=data["risk_percent"],
                stop_loss_price=data["stop_loss_price"],
                warnings=data.get("warnings", []),
                take_profit_price=data.get("take_profit_price"),
                reward_risk_ratio=data.get("reward_risk_ratio"),
                portfolio_var_95=data.get("portfolio_var_95"),
                portfolio_cvar_99=data.get("portfolio_cvar_99"),
                portfolio_cdar_95=data.get("portfolio_cdar_95"),
            )

            await repo.create(record)
            migrated += 1

            if migrated % 50 == 0:
                logger.info(f"Migrated {migrated} records...")

        except Exception as e:
            logger.opt(exception=True).error(f"Line {line_num} failed: {e}")
            failed += 1

    if migrated > 0:
        backup_path = await asyncio.to_thread(_backup_file, jsonl_path)
        logger.info(f"Backed up to {backup_path}")

    return (migrated, skipped, failed)


if __name__ == "__main__":
    migrated, skipped, failed = asyncio.run(migrate_risk_audit())
    logger.info(f"Migration complete: {migrated} migrated, {skipped} skipped, {failed} failed")
