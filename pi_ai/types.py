"""
Content types for pi-ai message blocks.

These Pydantic models represent the various content types that can appear
in LLM messages: text, thinking/reasoning, images, and tool calls.
"""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class TextContent(BaseModel):
    """Text content block in a message."""

    type: Literal["text"] = "text"
    text: str
    text_signature: Optional[str] = Field(
        default=None, description="For OpenAI responses metadata"
    )


class ThinkingContent(BaseModel):
    """Thinking/reasoning content block."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: Optional[str] = Field(
        default=None, description="Provider-specific signature"
    )
    redacted: bool = Field(default=False, description="Safety filter redaction flag")


class ImageContent(BaseModel):
    """Image content block."""

    type: Literal["image"] = "image"
    data: str = Field(description="Base64-encoded image data")
    mime_type: str = Field(description="e.g., 'image/jpeg', 'image/png'")


class ToolCall(BaseModel):
    """Tool call from the assistant."""

    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: Optional[str] = Field(default=None, description="Google-specific")


# Union type for all content blocks
ContentBlock = Union[TextContent, ThinkingContent, ImageContent, ToolCall]


# -----------------------------------------------------------------------------
# Usage Types
# -----------------------------------------------------------------------------

class Usage(BaseModel):
    """Token usage tracking for API calls."""

    class Cost(BaseModel):
        """Cost breakdown for token usage."""

        input: float
        output: float
        cache_read: float = Field(default=0.0)
        cache_write: float = Field(default=0.0)
        total: float

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = Field(default=0)
    cache_write_tokens: int = Field(default=0)
    total_tokens: int
    cost: Cost


# -----------------------------------------------------------------------------
# Stop Reason
# -----------------------------------------------------------------------------

StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


# -----------------------------------------------------------------------------
# Message Types
# -----------------------------------------------------------------------------

class UserMessage(BaseModel):
    """Message from the user."""

    role: Literal["user"] = "user"
    content: Union[str, list[Union[TextContent, ImageContent]]]
    timestamp: int  # Unix timestamp in milliseconds


class AssistantMessage(BaseModel):
    """Message from the assistant/LLM."""

    role: Literal["assistant"] = "assistant"
    content: list[Union[TextContent, ThinkingContent, ToolCall]]
    api: str
    provider: str
    model: str
    response_id: Optional[str] = Field(default=None)
    usage: Usage
    stop_reason: StopReason
    error_message: Optional[str] = Field(default=None)
    timestamp: int


class ToolResultMessage(BaseModel):
    """Result from a tool execution."""

    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[Union[TextContent, ImageContent]]
    details: Optional[Any] = Field(default=None)
    is_error: bool
    timestamp: int


# Union type for all messages
Message = Union[UserMessage, AssistantMessage, ToolResultMessage]