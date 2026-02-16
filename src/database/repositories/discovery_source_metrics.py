"""Discovery source metrics repository."""

from datetime import UTC, date, datetime, timedelta

from loguru import logger
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.daemon.state.models import DiscoverySourceMetrics
from src.database.models import DiscoverySourceMetricsORM
from src.database.repositories.base import BaseRepository


class DiscoverySourceMetricsRepository(BaseRepository[DiscoverySourceMetrics]):
    """Repository for discovery source metrics."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository.

        Args:
            session: Database session
        """
        super().__init__(session, DiscoverySourceMetrics, DiscoverySourceMetricsORM)

    async def create_or_update_daily_metrics(
        self, source_type: str, measurement_date: date, metrics: DiscoverySourceMetrics
    ) -> DiscoverySourceMetrics:
        """Create or update daily metrics for a source.

        Args:
            source_type: Discovery source type
            measurement_date: Date of measurement
            metrics: Metrics to persist

        Returns:
            Created or updated metrics
        """
        stmt = select(DiscoverySourceMetricsORM).where(
            and_(
                DiscoverySourceMetricsORM.source_type == source_type,
                DiscoverySourceMetricsORM.measurement_date == measurement_date,
            )
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            for key, value in metrics.model_dump(exclude={"source_type", "measurement_date"}).items():
                setattr(existing, key, value)
            await self.session.flush()
            logger.debug(f"Updated metrics for {source_type} on {measurement_date}")
        else:
            orm = DiscoverySourceMetricsORM(
                source_type=source_type,
                measurement_date=measurement_date,
                **metrics.model_dump(exclude={"source_type", "measurement_date"}),
            )
            self.session.add(orm)
            await self.session.flush()
            existing = orm
            logger.debug(f"Created metrics for {source_type} on {measurement_date}")

        return self._to_domain(existing)

    async def get_latest_metrics(
        self, source_type: str | None = None, days_back: int = 30
    ) -> list[DiscoverySourceMetrics]:
        """Get latest metrics for sources.

        Args:
            source_type: Optional source type filter
            days_back: Days to look back

        Returns:
            List of metrics ordered by date descending
        """
        cutoff_date = datetime.now(UTC).date() - timedelta(days=days_back)

        stmt = select(DiscoverySourceMetricsORM).where(
            DiscoverySourceMetricsORM.measurement_date >= cutoff_date
        )

        if source_type:
            stmt = stmt.where(DiscoverySourceMetricsORM.source_type == source_type)

        stmt = stmt.order_by(desc(DiscoverySourceMetricsORM.measurement_date))

        result = await self.session.execute(stmt)
        orms = result.scalars().all()

        return [self._to_domain(orm) for orm in orms]

    async def get_source_performance_ranking(
        self, metric: str = "f1_score", days_back: int = 30
    ) -> dict[str, float]:
        """Get source performance ranking by metric.

        Args:
            metric: Metric to rank by (f1_score, precision_score, avg_7d_return, etc.)
            days_back: Days to look back

        Returns:
            Dict mapping source_type to average metric value
        """
        cutoff_date = datetime.now(UTC).date() - timedelta(days=days_back)

        metric_column = getattr(DiscoverySourceMetricsORM, metric)

        stmt = (
            select(
                DiscoverySourceMetricsORM.source_type,
                func.avg(metric_column).label("avg_metric"),
            )
            .where(
                and_(
                    DiscoverySourceMetricsORM.measurement_date >= cutoff_date,
                    metric_column.is_not(None),
                )
            )
            .group_by(DiscoverySourceMetricsORM.source_type)
            .order_by(desc("avg_metric"))
        )

        result = await self.session.execute(stmt)
        rows = result.all()

        return {row.source_type: float(row.avg_metric) for row in rows}

    async def calculate_metrics_for_date(
        self, measurement_date: date, window_days: int = 7
    ) -> list[DiscoverySourceMetrics]:
        """Calculate metrics for all sources on a specific date.

        This is called by DiscoveryOutcomeTracker to compute daily metrics.

        Args:
            measurement_date: Date to calculate metrics for
            window_days: Window for rolling calculations

        Returns:
            List of calculated metrics per source
        """
        from src.database.repositories.discovery import DiscoveryHistoryRepository

        discovery_repo = DiscoveryHistoryRepository(self.session)

        all_sources = await discovery_repo.get_all_sources()

        metrics_list = []
        for source_type in all_sources:
            metrics = await self._calculate_source_metrics(
                source_type, measurement_date, window_days, discovery_repo
            )
            if metrics:
                metrics_list.append(metrics)

        return metrics_list

    async def _calculate_source_metrics(
        self,
        source_type: str,
        measurement_date: date,
        window_days: int,
        discovery_repo: object,
    ) -> DiscoverySourceMetrics | None:
        """Calculate metrics for a single source.

        Args:
            source_type: Source type
            measurement_date: Measurement date
            window_days: Calculation window
            discovery_repo: Discovery history repository

        Returns:
            Calculated metrics or None if insufficient data
        """
        positive_return_threshold = 0.02

        discoveries = await discovery_repo.get_discoveries_by_source(
            source_type, days_back=window_days, reference_date=measurement_date
        )

        if not discoveries:
            return None

        total_discoveries = len(discoveries)
        watchlist_additions = sum(1 for d in discoveries if d.added_to_watchlist)

        discoveries_with_7d = [d for d in discoveries if d.outcome_7d is not None]
        positive_7d = sum(
            1 for d in discoveries_with_7d if d.outcome_7d and d.outcome_7d > positive_return_threshold
        )

        discoveries_with_30d = [d for d in discoveries if d.outcome_30d is not None]
        positive_30d = sum(
            1 for d in discoveries_with_30d if d.outcome_30d and d.outcome_30d > positive_return_threshold
        )

        avg_7d = None
        median_7d = None
        if discoveries_with_7d:
            returns_7d = [d.outcome_7d for d in discoveries_with_7d if d.outcome_7d is not None]
            avg_7d = sum(returns_7d) / len(returns_7d) if returns_7d else None
            if returns_7d:
                sorted_returns = sorted(returns_7d)
                mid = len(sorted_returns) // 2
                median_7d = sorted_returns[mid]

        avg_30d = None
        median_30d = None
        if discoveries_with_30d:
            returns_30d = [d.outcome_30d for d in discoveries_with_30d if d.outcome_30d is not None]
            avg_30d = sum(returns_30d) / len(returns_30d) if returns_30d else None
            if returns_30d:
                sorted_returns = sorted(returns_30d)
                mid = len(sorted_returns) // 2
                median_30d = sorted_returns[mid]

        added_and_positive = sum(
            1
            for d in discoveries
            if d.added_to_watchlist and d.outcome_7d is not None and d.outcome_7d > positive_return_threshold
        )
        skipped_and_positive = sum(
            1
            for d in discoveries
            if not d.added_to_watchlist
            and d.outcome_7d is not None
            and d.outcome_7d > positive_return_threshold
        )

        precision_score = None
        recall_score = None
        f1_score = None

        if watchlist_additions > 0:
            precision_score = added_and_positive / watchlist_additions

        total_positive = added_and_positive + skipped_and_positive
        if total_positive > 0:
            recall_score = added_and_positive / total_positive

        if precision_score is not None and recall_score is not None and (precision_score + recall_score) > 0:
            f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)

        return DiscoverySourceMetrics(
            source_type=source_type,
            measurement_date=measurement_date,
            total_discoveries=total_discoveries,
            watchlist_additions=watchlist_additions,
            signal_conversions=0,
            discoveries_with_7d_outcome=len(discoveries_with_7d),
            positive_7d_outcomes=positive_7d,
            avg_7d_return=avg_7d,
            median_7d_return=median_7d,
            discoveries_with_30d_outcome=len(discoveries_with_30d),
            positive_30d_outcomes=positive_30d,
            avg_30d_return=avg_30d,
            median_30d_return=median_30d,
            precision_score=precision_score,
            recall_score=recall_score,
            f1_score=f1_score,
            false_positives=watchlist_additions - added_and_positive,
            false_negatives=skipped_and_positive,
        )

    def _to_domain(self, orm: DiscoverySourceMetricsORM) -> DiscoverySourceMetrics:
        """Convert ORM to domain model.

        Args:
            orm: ORM instance

        Returns:
            Domain model
        """
        return DiscoverySourceMetrics(
            source_type=orm.source_type,
            measurement_date=orm.measurement_date,
            total_discoveries=orm.total_discoveries,
            watchlist_additions=orm.watchlist_additions,
            signal_conversions=orm.signal_conversions,
            discoveries_with_7d_outcome=orm.discoveries_with_7d_outcome,
            positive_7d_outcomes=orm.positive_7d_outcomes,
            avg_7d_return=float(orm.avg_7d_return) if orm.avg_7d_return else None,
            median_7d_return=float(orm.median_7d_return) if orm.median_7d_return else None,
            discoveries_with_30d_outcome=orm.discoveries_with_30d_outcome,
            positive_30d_outcomes=orm.positive_30d_outcomes,
            avg_30d_return=float(orm.avg_30d_return) if orm.avg_30d_return else None,
            median_30d_return=float(orm.median_30d_return) if orm.median_30d_return else None,
            precision_score=float(orm.precision_score) if orm.precision_score else None,
            recall_score=float(orm.recall_score) if orm.recall_score else None,
            f1_score=float(orm.f1_score) if orm.f1_score else None,
            false_positives=orm.false_positives,
            false_negatives=orm.false_negatives,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "DiscoverySourceMetricsRepository()"
