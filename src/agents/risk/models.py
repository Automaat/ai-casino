"""Risk management data models."""

from datetime import datetime

from pydantic import BaseModel, Field

from src.metrics.portfolio_var import PortfolioVaRResult
from src.strategies.signal import Signal


class AccountInfo(BaseModel):
    """Account information for risk calculations."""

    balance: float
    available_cash: float
    positions: dict[str, float]
    total_exposure: float


class PositionSizeCalculation(BaseModel):
    """Position sizing result."""

    recommended_shares: int
    position_value: float
    risk_amount: float
    risk_percent: float
    reasoning: str


class TrailingStopConfig(BaseModel):
    """Trailing stop-loss configuration."""

    enabled: bool
    trail_percent: float
    activation_percent: float


class StopLossCalculation(BaseModel):
    """Stop-loss calculation result."""

    stop_loss_price: float
    stop_loss_percent: float
    risk_per_share: float
    max_loss_amount: float
    methodology: str
    trailing_stop: TrailingStopConfig | None = None


class RiskValidation(BaseModel):
    """Risk validation result."""

    approved: bool
    risk_score: float
    risk_level: str
    warnings: list[str]
    constraints_met: dict[str, bool]
    reasoning: str


class PortfolioVaRConfig(BaseModel):
    """Configuration for portfolio-level VaR limits."""

    enabled: bool = False
    max_var_95: float = Field(default=0.03, ge=0.001, le=0.20)
    max_cvar_99: float = Field(default=0.05, ge=0.001, le=0.30)
    lookback_days: int = Field(default=90, ge=20, le=365)
    adaptive_stop_loss: bool = True
    cdar_stop_threshold: float = Field(default=0.10, ge=0.01, le=0.50)
    atr_multiplier_min: float = Field(default=1.0, ge=0.5, le=2.0)


class PortfolioRiskReport(BaseModel):
    """Daily portfolio risk report."""

    date: str
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    cdar_95: float
    max_drawdown: float
    portfolio_volatility: float
    current_exposure_percent: float
    num_positions: int
    var_limit_breached: bool
    cvar_limit_breached: bool
    risk_status: str


class TakeProfitCalculation(BaseModel):
    """Take-profit calculation result."""

    take_profit_price: float
    take_profit_percent: float
    potential_profit_per_share: float
    reward_risk_ratio: float
    methodology: str


class RiskAssessment(BaseModel):
    """Complete risk management assessment."""

    symbol: str
    action: Signal
    current_price: float
    account_info: AccountInfo
    position_sizing: PositionSizeCalculation
    stop_loss: StopLossCalculation
    validation: RiskValidation
    confidence: float
    take_profit: TakeProfitCalculation | None = None
    reward_risk_ratio: float | None = None
    portfolio_var: PortfolioVaRResult | None = None


class RiskAuditRecord(BaseModel):
    """Risk audit log record."""

    id: str | None = None
    timestamp: datetime
    symbol: str
    action: Signal
    current_price: float

    approved: bool
    risk_level: str
    risk_score: float
    confidence: float

    recommended_shares: int
    position_value: float
    risk_amount: float
    risk_percent: float

    stop_loss_price: float
    warnings: list[str] = Field(default_factory=list)

    take_profit_price: float | None = None
    reward_risk_ratio: float | None = None

    portfolio_var_95: float | None = None
    portfolio_cvar_99: float | None = None
    portfolio_cdar_95: float | None = None

    created_at: datetime | None = None
