"""Typed models for LLM message structures."""

from typing import Any

from pydantic import BaseModel, Field


class ToolUseContent(BaseModel):
    """Anthropic tool use content block."""

    type: str = Field(default="tool_use", description="Content type")
    id: str = Field(description="Tool call ID")
    name: str = Field(description="Tool name")
    input: dict[str, Any] = Field(description="Tool input arguments")

    def __repr__(self) -> str:
        """String representation."""
        return f"ToolUseContent(id={self.id}, name={self.name})"


class ToolResultContent(BaseModel):
    """Anthropic tool result content block."""

    type: str = Field(default="tool_result", description="Content type")
    tool_use_id: str = Field(description="ID of tool use this result is for")
    content: str = Field(description="Tool execution result")

    def __repr__(self) -> str:
        """String representation."""
        return f"ToolResultContent(tool_use_id={self.tool_use_id})"


class OpenAIToolFunction(BaseModel):
    """OpenAI tool function definition."""

    name: str = Field(description="Function name")
    arguments: str = Field(description="JSON-encoded arguments")

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAIToolFunction(name={self.name})"


class OpenAIToolCall(BaseModel):
    """OpenAI tool call structure."""

    id: str = Field(description="Tool call ID")
    type: str = Field(default="function", description="Tool call type")
    function: OpenAIToolFunction = Field(description="Function details")

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAIToolCall(id={self.id}, function={self.function.name})"


class AnthropicAssistantMessage(BaseModel):
    """Anthropic assistant message with tool uses."""

    role: str = Field(default="assistant", description="Message role")
    content: list[ToolUseContent] = Field(description="Tool use content blocks")

    def __repr__(self) -> str:
        """String representation."""
        return f"AnthropicAssistantMessage(tools={len(self.content)})"


class OpenAIAssistantMessage(BaseModel):
    """OpenAI assistant message with tool calls."""

    role: str = Field(default="assistant", description="Message role")
    content: None = Field(default=None, description="No text content for tool calls")
    tool_calls: list[OpenAIToolCall] = Field(description="Tool calls to execute")

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAIAssistantMessage(tool_calls={len(self.tool_calls)})"


class AnthropicToolResultMessage(BaseModel):
    """Anthropic tool result message."""

    role: str = Field(default="user", description="Message role (user for tool results)")
    content: list[ToolResultContent] = Field(description="Tool result content blocks")

    def __repr__(self) -> str:
        """String representation."""
        return f"AnthropicToolResultMessage(results={len(self.content)})"


class OpenAIToolResultMessage(BaseModel):
    """OpenAI tool result message."""

    role: str = Field(default="tool", description="Message role (tool for results)")
    tool_call_id: str = Field(description="ID of tool call this result is for")
    content: str = Field(description="Tool execution result")

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAIToolResultMessage(tool_call_id={self.tool_call_id})"
