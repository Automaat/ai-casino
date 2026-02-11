"""WorkflowContext composing all stage outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.workflows.models.account import AccountInfoOutput
    from src.workflows.models.analysis import AnalysisOutput
    from src.workflows.models.backtest import BacktestValidationOutput
    from src.workflows.models.data_fetch import FetchDataOutput
    from src.workflows.models.decision import DecisionOutput
    from src.workflows.models.execution import TradeExecutionOutput
    from src.workflows.models.risk import RiskAssessmentOutput
    from src.workflows.models.strategy import StrategySelectionOutput


class WorkflowContext(BaseModel):
    """Composed workflow context from all stage outputs."""

    data: FetchDataOutput
    account: AccountInfoOutput
    strategy: StrategySelectionOutput
    backtest: BacktestValidationOutput
    analysis: AnalysisOutput
    decision: DecisionOutput
    risk: RiskAssessmentOutput
    execution: TradeExecutionOutput
    sector_rotation_context: str | None = None
    earnings_context: str | None = None
    peer_analysis_context: str | None = None
    game_plan_context: str | None = None
    position_context: dict[str, object] | None = None
    degradation_context: DegradationContext | None = None
    target_portfolio_weight: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def all_warnings(self) -> list[str]:
        """Aggregate warnings from all stages."""
        warnings = []
        warnings.extend(self.data.warnings)
        warnings.extend(self.account.warnings)
        warnings.extend(self.backtest.warnings)
        warnings.extend(self.analysis.warnings)
        warnings.extend(self.execution.warnings)
        return warnings
