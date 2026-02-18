"""Workflow execution configuration."""

from pydantic import BaseModel, Field


class WorkflowConfigDaemon(BaseModel):
    """Workflow execution configuration for daemon."""

    use_ensemble: bool = Field(default=False, description="Use ensemble strategy for technical analysis")
    use_meta_agent: bool = Field(default=True, description="Use meta-agent for dynamic strategy selection")
    trump_mode: bool = Field(default=False, description="Enable Trump social media analysis")
