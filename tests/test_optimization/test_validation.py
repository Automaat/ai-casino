"""Tests for walk-forward validation module."""

from datetime import datetime

import pytest

from src.optimization.validation import ValidationFold, ValidationResult, WalkForwardValidator


class TestValidationFold:
    """Tests for ValidationFold."""

    def test_fold_creation(self):
        """Test creating a validation fold."""
        fold = ValidationFold(
            fold_number=1,
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 6, 30),
            test_start=datetime(2023, 7, 1),
            test_end=datetime(2023, 12, 31),
        )

        assert fold.fold_number == 1
        assert fold.train_start == datetime(2023, 1, 1)
        assert fold.test_end == datetime(2023, 12, 31)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_result_creation(self):
        """Test creating a validation result."""
        folds = [
            ValidationFold(
                fold_number=1,
                train_start=datetime(2023, 1, 1),
                train_end=datetime(2023, 6, 30),
                test_start=datetime(2023, 7, 1),
                test_end=datetime(2023, 12, 31),
            )
        ]

        result = ValidationResult(
            folds=folds,
            metrics_avg={"sharpe_ratio": 1.2, "total_return": 0.15},
            metrics_std={"sharpe_ratio": 0.3, "total_return": 0.05},
            fold_metrics=[{"sharpe_ratio": 1.2, "total_return": 0.15}],
        )

        assert len(result.folds) == 1
        assert result.metrics_avg["sharpe_ratio"] == 1.2

    def test_repr(self):
        """Test string representation."""
        result = ValidationResult(
            folds=[],
            metrics_avg={"sharpe_ratio": 1.5},
            metrics_std={"sharpe_ratio": 0.2},
            fold_metrics=[],
        )

        repr_str = repr(result)
        assert "folds=0" in repr_str
        assert "sharpe=1.50" in repr_str


class TestWalkForwardValidator:
    """Tests for WalkForwardValidator."""

    def test_init_defaults(self):
        """Test default initialization."""
        validator = WalkForwardValidator()

        assert validator.n_splits == 5
        assert validator.train_ratio == 0.7
        assert validator.gap_days == 0

    def test_init_custom(self):
        """Test custom initialization."""
        validator = WalkForwardValidator(n_splits=3, train_ratio=0.8, gap_days=5)

        assert validator.n_splits == 3
        assert validator.train_ratio == 0.8
        assert validator.gap_days == 5

    def test_invalid_splits(self):
        """Test error on invalid splits."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            WalkForwardValidator(n_splits=1)

    def test_invalid_train_ratio(self):
        """Test error on invalid train ratio."""
        with pytest.raises(ValueError, match="train_ratio must be between"):
            WalkForwardValidator(train_ratio=0.3)

        with pytest.raises(ValueError, match="train_ratio must be between"):
            WalkForwardValidator(train_ratio=0.95)

    def test_generate_folds(self):
        """Test fold generation."""
        validator = WalkForwardValidator(n_splits=3, train_ratio=0.7, gap_days=0)

        folds = validator.generate_folds(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        assert len(folds) == 3
        assert folds[0].fold_number == 1
        assert folds[2].fold_number == 3

        # Train end should be before test start
        for fold in folds:
            assert fold.train_end <= fold.test_start

    def test_generate_folds_with_gap(self):
        """Test fold generation with gap days."""
        validator = WalkForwardValidator(n_splits=2, train_ratio=0.7, gap_days=7)

        folds = validator.generate_folds(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        for fold in folds:
            gap = (fold.test_start - fold.train_end).days
            assert gap >= 7

    def test_validate(self):
        """Test validation run."""
        validator = WalkForwardValidator(n_splits=2, train_ratio=0.7)

        def mock_objective(train_start, train_end, test_start, test_end):
            return {"sharpe_ratio": 1.0, "total_return": 0.1}

        result = validator.validate(
            objective_fn=mock_objective,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        assert len(result.folds) == 2
        assert len(result.fold_metrics) == 2
        assert result.metrics_avg["sharpe_ratio"] == 1.0

    def test_repr(self):
        """Test string representation."""
        validator = WalkForwardValidator(n_splits=5, train_ratio=0.7, gap_days=3)

        repr_str = repr(validator)
        assert "splits=5" in repr_str
        assert "train_ratio=0.7" in repr_str
        assert "gap=3d" in repr_str
