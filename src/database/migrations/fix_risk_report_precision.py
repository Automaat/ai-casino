"""Fix risk_report_records.current_exposure_percent precision.

This migration fixes a numeric overflow issue where current_exposure_percent
was defined as DECIMAL(5,4) which can only hold values up to 9.9999.
The field needs DECIMAL(8,4) to support exposure percentages up to 9999.9999.

Usage:
    python -m src.database.migrations.fix_risk_report_precision
"""

import asyncio

from loguru import logger
from sqlalchemy import text

from src.database.engine import DatabaseEngine

OLD_PRECISION = 5
NEW_PRECISION = 8
SCALE = 4


async def migrate() -> None:
    """Apply migration to fix current_exposure_percent precision."""
    engine = DatabaseEngine()

    async with engine.engine.begin() as conn:
        # Check current precision
        result = await conn.execute(
            text(
                """
                SELECT numeric_precision, numeric_scale
                FROM information_schema.columns
                WHERE table_name = 'risk_report_records'
                AND column_name = 'current_exposure_percent'
                """
            )
        )
        row = result.fetchone()

        if row and row[0] == OLD_PRECISION:
            logger.info(
                f"Applying migration: changing current_exposure_percent from "
                f"({OLD_PRECISION},{SCALE}) to ({NEW_PRECISION},{SCALE})"
            )
            await conn.execute(
                text(
                    f"""
                    ALTER TABLE risk_report_records
                    ALTER COLUMN current_exposure_percent TYPE numeric({NEW_PRECISION},{SCALE})
                    """
                )
            )
            logger.info("Migration completed successfully")
        else:
            logger.info(f"Migration already applied or unexpected precision: {row[0] if row else 'unknown'}")


async def main() -> None:
    """Run migration."""
    try:
        await migrate()
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
