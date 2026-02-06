"""TearSheet repository for database operations."""

from decimal import Decimal

from loguru import logger
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TearSheetORM
from src.metrics.tracker import TearSheet


class TearSheetRepository:
    """Repository for tearsheet database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository.

        Args:
            session: AsyncIO database session
        """
        self.session = session

    async def create(self, tearsheet: TearSheet) -> TearSheet:
        """Create a new tearsheet.

        Args:
            tearsheet: TearSheet to create

        Returns:
            Created TearSheet with database ID
        """
        logger.debug(f"Creating tearsheet for {tearsheet.symbol}")

        orm = TearSheetORM(
            symbol=tearsheet.symbol,
            start_date=tearsheet.start_date,
            end_date=tearsheet.end_date,
            cagr=Decimal(str(tearsheet.cagr)) if tearsheet.cagr is not None else None,
            sharpe_ratio=Decimal(str(tearsheet.sharpe_ratio)) if tearsheet.sharpe_ratio is not None else None,
            sortino_ratio=Decimal(str(tearsheet.sortino_ratio))
            if tearsheet.sortino_ratio is not None
            else None,
            calmar_ratio=Decimal(str(tearsheet.calmar_ratio)) if tearsheet.calmar_ratio is not None else None,
            max_drawdown=Decimal(str(tearsheet.max_drawdown)) if tearsheet.max_drawdown is not None else None,
            max_drawdown_duration_days=tearsheet.max_drawdown_duration_days,
            volatility_annual=(
                Decimal(str(tearsheet.volatility_annual)) if tearsheet.volatility_annual is not None else None
            ),
            win_rate=Decimal(str(tearsheet.win_rate)) if tearsheet.win_rate is not None else None,
            profit_factor=Decimal(str(tearsheet.profit_factor))
            if tearsheet.profit_factor is not None
            else None,
            avg_win=Decimal(str(tearsheet.avg_win)) if tearsheet.avg_win is not None else None,
            avg_loss=Decimal(str(tearsheet.avg_loss)) if tearsheet.avg_loss is not None else None,
            best_day=Decimal(str(tearsheet.best_day)) if tearsheet.best_day is not None else None,
            worst_day=Decimal(str(tearsheet.worst_day)) if tearsheet.worst_day is not None else None,
            monthly_returns=tearsheet.monthly_returns,
            benchmark_symbol=tearsheet.benchmark_symbol,
            benchmark_cagr=(
                Decimal(str(tearsheet.benchmark_cagr)) if tearsheet.benchmark_cagr is not None else None
            ),
            benchmark_sharpe=(
                Decimal(str(tearsheet.benchmark_sharpe)) if tearsheet.benchmark_sharpe is not None else None
            ),
            alpha=Decimal(str(tearsheet.alpha)) if tearsheet.alpha is not None else None,
            beta=Decimal(str(tearsheet.beta)) if tearsheet.beta is not None else None,
            html_report_path=tearsheet.html_report_path,
            generated_at=tearsheet.generated_at,
        )

        self.session.add(orm)
        await self.session.commit()
        await self.session.refresh(orm)

        logger.info(f"Created tearsheet {orm.id} for {tearsheet.symbol}")
        return self._to_pydantic(orm)

    async def get_by_id(self, tearsheet_id: str) -> TearSheet | None:
        """Get tearsheet by ID.

        Args:
            tearsheet_id: Tearsheet UUID

        Returns:
            TearSheet or None if not found
        """
        stmt = select(TearSheetORM).where(TearSheetORM.id == tearsheet_id)
        result = await self.session.execute(stmt)
        orm = result.scalar_one_or_none()

        if orm is None:
            logger.debug(f"Tearsheet {tearsheet_id} not found")
            return None

        return self._to_pydantic(orm)

    async def get_by_symbol(self, symbol: str) -> list[TearSheet]:
        """Get all tearsheets for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of TearSheet objects ordered by generated_at desc
        """
        stmt = (
            select(TearSheetORM)
            .where(TearSheetORM.symbol == symbol)
            .order_by(desc(TearSheetORM.generated_at))
        )
        result = await self.session.execute(stmt)
        orms = result.scalars().all()

        logger.debug(f"Found {len(orms)} tearsheets for {symbol}")
        return [self._to_pydantic(orm) for orm in orms]

    async def get_latest(self, symbol: str) -> TearSheet | None:
        """Get latest tearsheet for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            TearSheet or None if not found
        """
        stmt = (
            select(TearSheetORM)
            .where(TearSheetORM.symbol == symbol)
            .order_by(desc(TearSheetORM.generated_at))
            .limit(1)
        )
        result = await self.session.execute(stmt)
        orm = result.scalar_one_or_none()

        if orm is None:
            logger.debug(f"No tearsheet found for {symbol}")
            return None

        return self._to_pydantic(orm)

    async def get_all(self) -> list[TearSheet]:
        """Get all tearsheets.

        Returns:
            List of TearSheet objects ordered by generated_at desc
        """
        stmt = select(TearSheetORM).order_by(desc(TearSheetORM.generated_at))
        result = await self.session.execute(stmt)
        orms = result.scalars().all()

        logger.debug(f"Found {len(orms)} tearsheets")
        return [self._to_pydantic(orm) for orm in orms]

    def _to_pydantic(self, orm: TearSheetORM) -> TearSheet:
        """Convert ORM to Pydantic model.

        Args:
            orm: TearSheetORM object

        Returns:
            TearSheet Pydantic model
        """
        return TearSheet(
            id=str(orm.id),
            symbol=orm.symbol,
            start_date=orm.start_date,
            end_date=orm.end_date,
            cagr=float(orm.cagr) if orm.cagr is not None else None,
            sharpe_ratio=float(orm.sharpe_ratio) if orm.sharpe_ratio is not None else None,
            sortino_ratio=float(orm.sortino_ratio) if orm.sortino_ratio is not None else None,
            calmar_ratio=float(orm.calmar_ratio) if orm.calmar_ratio is not None else None,
            max_drawdown=float(orm.max_drawdown) if orm.max_drawdown is not None else None,
            max_drawdown_duration_days=orm.max_drawdown_duration_days,
            volatility_annual=float(orm.volatility_annual) if orm.volatility_annual is not None else None,
            win_rate=float(orm.win_rate) if orm.win_rate is not None else None,
            profit_factor=float(orm.profit_factor) if orm.profit_factor is not None else None,
            avg_win=float(orm.avg_win) if orm.avg_win is not None else None,
            avg_loss=float(orm.avg_loss) if orm.avg_loss is not None else None,
            best_day=float(orm.best_day) if orm.best_day is not None else None,
            worst_day=float(orm.worst_day) if orm.worst_day is not None else None,
            monthly_returns=orm.monthly_returns,
            benchmark_symbol=orm.benchmark_symbol,
            benchmark_cagr=float(orm.benchmark_cagr) if orm.benchmark_cagr is not None else None,
            benchmark_sharpe=float(orm.benchmark_sharpe) if orm.benchmark_sharpe is not None else None,
            alpha=float(orm.alpha) if orm.alpha is not None else None,
            beta=float(orm.beta) if orm.beta is not None else None,
            html_report_path=orm.html_report_path,
            generated_at=orm.generated_at,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "TearSheetRepository()"
