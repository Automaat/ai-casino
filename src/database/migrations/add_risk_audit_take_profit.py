"""Add take_profit_price and reward_risk_ratio columns to risk_audit table.

Usage:
    python -m src.database.migrations.add_risk_audit_take_profit

Rollback (manual):
    ALTER TABLE risk_audit DROP COLUMN IF EXISTS take_profit_price;
    ALTER TABLE risk_audit DROP COLUMN IF EXISTS reward_risk_ratio;
"""

import asyncio

from loguru import logger
from sqlalchemy import text

from src.database.engine import DatabaseEngine


async def migrate() -> None:
    """Add take-profit and reward:risk ratio columns to risk_audit."""
    engine = DatabaseEngine()

    async with engine.engine.begin() as conn:
        result = await conn.execute(
            text(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'risk_audit' AND column_name = 'take_profit_price'
                """
            )
        )
        if result.scalar():
            logger.info("take_profit_price column already exists, skipping")
            return

        await conn.execute(text("ALTER TABLE risk_audit ADD COLUMN take_profit_price DECIMAL(12, 4)"))
        await conn.execute(text("ALTER TABLE risk_audit ADD COLUMN reward_risk_ratio DECIMAL(5, 2)"))

        logger.info("Added take_profit_price and reward_risk_ratio to risk_audit")

    await engine.engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())
