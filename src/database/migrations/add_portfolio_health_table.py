"""Add portfolio_health_reports table for portfolio health check persistence.

Usage:
    python -m src.database.migrations.add_portfolio_health_table
"""

import asyncio

from loguru import logger
from sqlalchemy import text

from src.database.engine import DatabaseEngine


async def migrate() -> None:
    """Apply migration to create portfolio_health_reports table."""
    engine = DatabaseEngine()

    async with engine.engine.begin() as conn:
        result = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'portfolio_health_reports'
                )
                """
            )
        )
        exists = result.scalar()

        if exists:
            logger.info("portfolio_health_reports table already exists, skipping")
            return

        await conn.execute(
            text(
                """
                CREATE TABLE portfolio_health_reports (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    total_positions INTEGER NOT NULL,
                    portfolio_value DECIMAL(16, 4) NOT NULL,
                    cash_percent DECIMAL(8, 4) NOT NULL,
                    max_concentration_percent DECIMAL(8, 4) NOT NULL,
                    max_concentration_symbol VARCHAR(20) NOT NULL,
                    total_pnl_percent DECIMAL(8, 4) NOT NULL,
                    biggest_drawdown_symbol VARCHAR(20),
                    biggest_drawdown_percent DECIMAL(8, 4) NOT NULL,
                    health_status VARCHAR(20) NOT NULL,
                    recommendations JSONB NOT NULL DEFAULT '[]'::jsonb,
                    constraints JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                )
                """
            )
        )

        await conn.execute(
            text("CREATE INDEX idx_portfolio_health_timestamp ON portfolio_health_reports (timestamp)")
        )
        await conn.execute(
            text("CREATE INDEX idx_portfolio_health_status ON portfolio_health_reports (health_status)")
        )

        logger.info("Created portfolio_health_reports table with indexes")

    await engine.engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())
