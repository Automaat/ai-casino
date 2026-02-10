"""Account info stage output model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.risk import AccountInfo
from src.data.broker import BrokerPosition


class AccountInfoOutput(BaseModel):
    """Output from account info fetch stage."""

    account_info: AccountInfo | None
    broker_positions: dict[str, BrokerPosition] | None
    portfolio_value: float | None
    broker_api_failed: bool = False
    warnings: list[str] = Field(default_factory=list)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
