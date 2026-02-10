"""Backtest validation stage output model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.workflows.types import BacktestValidation


class BacktestValidationOutput(BaseModel):
    """Output from backtest validation stage."""

    backtest_validation: BacktestValidation | None
    warnings: list[str] = Field(default_factory=list)
