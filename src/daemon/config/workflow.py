"""Workflow execution configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class WorkflowConfigDaemon(BaseModel):
    """Workflow execution configuration for daemon."""

    analysis_pattern: Literal["sequential", "supervisor"] = Field(
        default="sequential",
        description=(
            "Workflow execution pattern: sequential (agents, 8-stage pipeline) "
            "or supervisor (workers, adaptive orchestration)"
        ),
    )
    use_ensemble: bool = Field(default=False, description="Use ensemble strategy for technical analysis")
    use_meta_agent: bool = Field(default=True, description="Use meta-agent for dynamic strategy selection")
    trump_mode: bool = Field(default=False, description="Enable Trump social media analysis")
