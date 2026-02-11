"""Typed models for LLM tool definitions."""

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    type: str = Field(description="Parameter type (string, number, boolean, etc.)")
    description: str = Field(description="Parameter description for LLM")
    enum: list[str] | None = Field(default=None, description="Valid enum values if applicable")

    def __repr__(self) -> str:
        """String representation."""
        enum_part = f", enum={self.enum}" if self.enum else ""
        return f"ToolParameter(type={self.type}{enum_part})"


class ToolParametersSchema(BaseModel):
    """Tool parameters schema definition."""

    type: str = Field(default="object", description="Schema type (always 'object' for tools)")
    properties: dict[str, ToolParameter] = Field(description="Parameter definitions keyed by name")
    required: list[str] = Field(default_factory=list, description="Required parameter names")

    def __repr__(self) -> str:
        """String representation."""
        return f"ToolParametersSchema(properties={list(self.properties.keys())}, required={self.required})"


class ToolFunction(BaseModel):
    """Tool function definition."""

    name: str = Field(description="Tool function name")
    description: str = Field(description="Tool function description for LLM")
    parameters: ToolParametersSchema = Field(description="Parameter schema")

    def __repr__(self) -> str:
        """String representation."""
        return f"ToolFunction(name={self.name})"


class ToolDefinition(BaseModel):
    """Complete tool definition for LLM function calling."""

    type: str = Field(default="function", description="Tool type (always 'function')")
    function: ToolFunction = Field(description="Function definition")

    def __repr__(self) -> str:
        """String representation."""
        return f"ToolDefinition(function={self.function.name})"
