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