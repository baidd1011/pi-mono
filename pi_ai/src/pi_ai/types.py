"""
Content types for pi-ai message blocks.

These Pydantic models represent the various content types that can appear
in LLM messages: text, thinking/reasoning, images, and tool calls.
"""

from typing import Any, Literal, Optional, Union, TypedDict

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


# -----------------------------------------------------------------------------
# Tool and Context Types (Task 4)
# -----------------------------------------------------------------------------

class Tool(BaseModel):
    """Tool definition for LLM function calling."""

    name: str
    description: str
    parameters: type[BaseModel]  # Pydantic model for schema

    def get_json_schema(self) -> dict:
        """Get JSON schema for the tool parameters."""
        return self.parameters.model_json_schema()


class Context(BaseModel):
    """Conversation context for LLM API calls."""

    system_prompt: Optional[str] = Field(default=None)
    messages: list[Message]
    tools: Optional[list[Tool]] = Field(default=None)


class Model(BaseModel):
    """Model configuration and metadata."""

    id: str
    name: str
    api: str
    provider: str
    base_url: str
    reasoning: bool
    input: list[Literal["text", "image"]]
    cost: Usage.Cost
    context_window: int
    max_tokens: int
    headers: Optional[dict[str, str]] = Field(default=None)


# Thinking level for reasoning models
ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]


# -----------------------------------------------------------------------------
# AssistantMessageEvent Types (Task 7)
# -----------------------------------------------------------------------------

class StartEvent(TypedDict):
    """Event emitted when a message stream starts."""

    type: Literal["start"]
    api: str
    provider: str
    model: str


class TextStartEvent(TypedDict):
    """Event emitted when a text content block starts."""

    type: Literal["textStart"]
    index: int


class TextDeltaEvent(TypedDict):
    """Event emitted for text content delta."""

    type: Literal["textDelta"]
    index: int
    delta: str


class TextEndEvent(TypedDict):
    """Event emitted when a text content block ends."""

    type: Literal["textEnd"]
    index: int
    text_signature: Optional[str]


class ThinkingStartEvent(TypedDict):
    """Event emitted when a thinking content block starts."""

    type: Literal["thinkingStart"]
    index: int


class ThinkingDeltaEvent(TypedDict):
    """Event emitted for thinking content delta."""

    type: Literal["thinkingDelta"]
    index: int
    delta: str


class ThinkingEndEvent(TypedDict):
    """Event emitted when a thinking content block ends."""

    type: Literal["thinkingEnd"]
    index: int
    thinking_signature: Optional[str]


class ToolCallStartEvent(TypedDict):
    """Event emitted when a tool call block starts."""

    type: Literal["toolCallStart"]
    index: int
    id: str
    name: str


class ToolCallDeltaEvent(TypedDict):
    """Event emitted for tool call argument delta."""

    type: Literal["toolCallDelta"]
    index: int
    id: str
    delta: str


class ToolCallEndEvent(TypedDict):
    """Event emitted when a tool call block ends."""

    type: Literal["toolCallEnd"]
    index: int
    id: str
    arguments: dict[str, Any]


class DoneEvent(TypedDict):
    """Event emitted when the message stream is complete."""

    type: Literal["done"]
    response_id: Optional[str]
    usage: Usage
    stop_reason: StopReason


class ErrorEvent(TypedDict):
    """Event emitted when an error occurs during streaming."""

    type: Literal["error"]
    error_message: str


# Union type for all assistant message events
AssistantMessageEvent = Union[
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
]