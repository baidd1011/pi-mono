# tests/test_types.py
"""Tests for pi_ai content types."""

import pytest
from pydantic import ValidationError


class TestTextContent:
    """Tests for TextContent model."""

    def test_text_content_creation(self):
        """TextContent can be created with required fields."""
        from pi_ai.types import TextContent

        content = TextContent(text="Hello world")
        assert content.type == "text"
        assert content.text == "Hello world"
        assert content.text_signature is None

    def test_text_content_with_signature(self):
        """TextContent can have optional signature."""
        from pi_ai.types import TextContent

        content = TextContent(text="Hello", text_signature="sig123")
        assert content.text_signature == "sig123"

    def test_text_content_requires_text(self):
        """TextContent requires text field."""
        from pi_ai.types import TextContent

        with pytest.raises(ValidationError):
            TextContent()


class TestThinkingContent:
    """Tests for ThinkingContent model."""

    def test_thinking_content_creation(self):
        """ThinkingContent can be created with required fields."""
        from pi_ai.types import ThinkingContent

        content = ThinkingContent(thinking="Reasoning steps...")
        assert content.type == "thinking"
        assert content.thinking == "Reasoning steps..."
        assert content.redacted is False

    def test_thinking_content_redacted(self):
        """ThinkingContent can be marked as redacted."""
        from pi_ai.types import ThinkingContent

        content = ThinkingContent(thinking="...", redacted=True)
        assert content.redacted is True

    def test_thinking_content_with_signature(self):
        """ThinkingContent can have optional signature."""
        from pi_ai.types import ThinkingContent

        content = ThinkingContent(thinking="...", thinking_signature="sig456")
        assert content.thinking_signature == "sig456"


class TestImageContent:
    """Tests for ImageContent model."""

    def test_image_content_creation(self):
        """ImageContent can be created with required fields."""
        from pi_ai.types import ImageContent

        content = ImageContent(data="base64data", mime_type="image/png")
        assert content.type == "image"
        assert content.data == "base64data"
        assert content.mime_type == "image/png"

    def test_image_content_requires_fields(self):
        """ImageContent requires both data and mime_type."""
        from pi_ai.types import ImageContent

        with pytest.raises(ValidationError):
            ImageContent(data="base64data")

        with pytest.raises(ValidationError):
            ImageContent(mime_type="image/png")


class TestToolCall:
    """Tests for ToolCall model."""

    def test_tool_call_creation(self):
        """ToolCall can be created with required fields."""
        from pi_ai.types import ToolCall

        call = ToolCall(id="call_123", name="get_weather", arguments={"city": "Beijing"})
        assert call.type == "toolCall"
        assert call.id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Beijing"}

    def test_tool_call_with_thought_signature(self):
        """ToolCall can have optional thought_signature."""
        from pi_ai.types import ToolCall

        call = ToolCall(
            id="call_456",
            name="search",
            arguments={"query": "test"},
            thought_signature="sig789"
        )
        assert call.thought_signature == "sig789"

    def test_tool_call_requires_fields(self):
        """ToolCall requires id, name, and arguments."""
        from pi_ai.types import ToolCall

        with pytest.raises(ValidationError):
            ToolCall(id="call_123")

        with pytest.raises(ValidationError):
            ToolCall(name="get_weather")


class TestContentBlock:
    """Tests for ContentBlock union type."""

    def test_content_block_union_text(self):
        """ContentBlock can be TextContent."""
        from pi_ai.types import ContentBlock, TextContent

        content: ContentBlock = TextContent(text="Hello")
        assert isinstance(content, TextContent)

    def test_content_block_union_thinking(self):
        """ContentBlock can be ThinkingContent."""
        from pi_ai.types import ContentBlock, ThinkingContent

        content: ContentBlock = ThinkingContent(thinking="...")
        assert isinstance(content, ThinkingContent)

    def test_content_block_union_image(self):
        """ContentBlock can be ImageContent."""
        from pi_ai.types import ContentBlock, ImageContent

        content: ContentBlock = ImageContent(data="abc", mime_type="image/png")
        assert isinstance(content, ImageContent)

    def test_content_block_union_tool_call(self):
        """ContentBlock can be ToolCall."""
        from pi_ai.types import ContentBlock, ToolCall

        content: ContentBlock = ToolCall(id="1", name="test", arguments={})
        assert isinstance(content, ToolCall)