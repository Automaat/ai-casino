"""Test migration script - verify new tables are created."""

import asyncio

from loguru import logger
from sqlalchemy import text

from src.database.engine import DatabaseEngine


async def test_migration() -> None:
    """Run migrations and verify new tables exist."""
    engine = DatabaseEngine()

    try:
        # Run migrations
        logger.info("Running migrations...")
        await engine.run_migrations()

        # Verify tables exist
        tables_to_check = [
            "daemon_metadata",
            "optimization_records",
            "rebalancing_records",
            "sector_rotation_records",
            "peer_analysis_records",
            "correlation_audit_records",
            "risk_report_records",
            "monte_carlo_records",
            "prefetch_records",
            "screening_records",
            "earnings_calendar_records",
            "profiling_records",
            "game_plan_records",
            "degradation_records",
            "active_discovery_candidates",
        ]

        async with engine.engine.connect() as conn:
            for table in tables_to_check:
                result = await conn.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = :table_name
                        )
                        """
                    ),
                    {"table_name": table},
                )
                exists = result.scalar()
                if exists:
                    logger.info(f"✓ Table '{table}' exists")
                else:
                    logger.error(f"✗ Table '{table}' missing")

        logger.info("Migration test complete!")

    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(test_migration())
