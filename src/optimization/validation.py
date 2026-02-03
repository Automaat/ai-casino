"""Walk-forward cross-validation for strategy backtesting."""

from datetime import datetime, timedelta

import pandas as pd
from loguru import logger
from pydantic import BaseModel

MIN_SPLITS = 2
MIN_TRAIN_RATIO = 0.5
MAX_TRAIN_RATIO = 0.9
MIN_TEST_DAYS = 5


class ValidationFold(BaseModel):
    """Single validation fold."""

    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


class ValidationResult(BaseModel):
    """Result from walk-forward validation."""

    folds: list[ValidationFold]
    metrics_avg: dict[str, float]
    metrics_std: dict[str, float]
    fold_metrics: list[dict[str, float]]

    def __repr__(self) -> str:
        """String representation."""
        sharpe_avg = self.metrics_avg.get("sharpe_ratio", 0)
        sharpe_std = self.metrics_std.get("sharpe_ratio", 0)
        return f"ValidationResult(folds={len(self.folds)}, sharpe={sharpe_avg:.2f}+/-{sharpe_std:.2f})"


class WalkForwardValidator:
    """Walk-forward cross-validation for backtesting."""

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        gap_days: int = 0,
    ) -> None:
        """Initialize walk-forward validator.

        Args:
            n_splits: Number of validation splits
            train_ratio: Ratio of data for training (0.0-1.0)
            gap_days: Gap between train and test periods
        """
        if n_splits < MIN_SPLITS:
            msg = f"n_splits must be >= {MIN_SPLITS}"
            raise ValueError(msg)
        if not MIN_TRAIN_RATIO <= train_ratio <= MAX_TRAIN_RATIO:
            msg = f"train_ratio must be between {MIN_TRAIN_RATIO} and {MAX_TRAIN_RATIO}"
            raise ValueError(msg)

        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap_days = gap_days

        logger.info(
            f"Initialized WalkForwardValidator: splits={n_splits}, train_ratio={train_ratio}, gap={gap_days}d"
        )

    def generate_folds(self, start_date: datetime, end_date: datetime) -> list[ValidationFold]:
        """Generate walk-forward validation folds.

        Args:
            start_date: Overall start date
            end_date: Overall end date

        Returns:
            List of ValidationFold
        """
        total_days = (end_date - start_date).days
        fold_size = total_days // self.n_splits

        train_days = int(fold_size * self.train_ratio)
        test_days = fold_size - train_days - self.gap_days

        if test_days < MIN_TEST_DAYS:
            msg = f"Test period too short ({test_days} days). Increase date range or reduce splits."
            raise ValueError(msg)

        folds = []
        for i in range(self.n_splits):
            fold_start = start_date + timedelta(days=i * fold_size)

            train_start = fold_start
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=self.gap_days)
            test_end = test_start + timedelta(days=test_days)

            test_end = min(test_end, end_date)

            folds.append(
                ValidationFold(
                    fold_number=i + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

        logger.debug(f"Generated {len(folds)} validation folds")
        return folds

    def validate(
        self,
        objective_fn,  # noqa: ANN001
        start_date: datetime,
        end_date: datetime,
    ) -> ValidationResult:
        """Run walk-forward validation.

        Args:
            objective_fn: Function(train_start, train_end, test_start, test_end) -> dict[str, float]
            start_date: Overall start date
            end_date: Overall end date

        Returns:
            ValidationResult with aggregated metrics
        """
        folds = self.generate_folds(start_date, end_date)
        fold_metrics: list[dict[str, float]] = []

        for fold in folds:
            logger.info(f"Running fold {fold.fold_number}/{len(folds)}")
            metrics = objective_fn(fold.train_start, fold.train_end, fold.test_start, fold.test_end)
            fold_metrics.append(metrics)

        metrics_df = pd.DataFrame(fold_metrics)
        metrics_avg = metrics_df.mean().to_dict()
        metrics_std = metrics_df.std().to_dict()

        result = ValidationResult(
            folds=folds,
            metrics_avg=metrics_avg,
            metrics_std=metrics_std,
            fold_metrics=fold_metrics,
        )

        logger.info(f"Walk-forward validation complete: {result}")
        return result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WalkForwardValidator(splits={self.n_splits}, "
            f"train_ratio={self.train_ratio}, gap={self.gap_days}d)"
        )
