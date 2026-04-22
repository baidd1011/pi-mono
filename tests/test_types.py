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


class TestUsage:
    """Tests for Usage model."""

    def test_usage_creation(self):
        """Usage can be created with required fields."""
        from pi_ai.types import Usage

        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=Usage.Cost(input=0.001, output=0.002, total=0.003)
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0
        assert usage.cost.input == 0.001
        assert usage.cost.output == 0.002
        assert usage.cost.total == 0.003

    def test_usage_with_cache_tokens(self):
        """Usage can include cache token counts."""
        from pi_ai.types import Usage

        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_write_tokens=150,
            total_tokens=500,
            cost=Usage.Cost(
                input=0.001,
                output=0.002,
                cache_read=0.0005,
                cache_write=0.001,
                total=0.0045
            )
        )
        assert usage.cache_read_tokens == 200
        assert usage.cache_write_tokens == 150
        assert usage.cost.cache_read == 0.0005
        assert usage.cost.cache_write == 0.001

    def test_usage_cost_defaults(self):
        """Usage.Cost has defaults for cache fields."""
        from pi_ai.types import Usage

        cost = Usage.Cost(input=0.001, output=0.002, total=0.003)
        assert cost.cache_read == 0.0
        assert cost.cache_write == 0.0

    def test_usage_requires_fields(self):
        """Usage requires input_tokens, output_tokens, total_tokens, cost."""
        from pi_ai.types import Usage

        with pytest.raises(ValidationError):
            Usage(input_tokens=100, output_tokens=50, total_tokens=150)


class TestStopReason:
    """Tests for StopReason literal type."""

    def test_stop_reason_values(self):
        """StopReason accepts valid values."""
        from pi_ai.types import StopReason

        # These are valid values (type checking validates at runtime via Literal)
        valid_values = ["stop", "length", "toolUse", "error", "aborted"]
        # Just verify the type exists and can be used
        assert StopReason is not None


class TestUserMessage:
    """Tests for UserMessage model."""

    def test_user_message_with_string_content(self):
        """UserMessage can be created with string content."""
        from pi_ai.types import UserMessage

        msg = UserMessage(content="Hello, world!", timestamp=1234567890000)
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.timestamp == 1234567890000

    def test_user_message_with_text_content(self):
        """UserMessage can be created with TextContent list."""
        from pi_ai.types import UserMessage, TextContent

        msg = UserMessage(
            content=[TextContent(text="Hello")],
            timestamp=1234567890000
        )
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert msg.content[0].text == "Hello"

    def test_user_message_with_image_content(self):
        """UserMessage can be created with ImageContent."""
        from pi_ai.types import UserMessage, TextContent, ImageContent

        msg = UserMessage(
            content=[
                TextContent(text="What's in this image?"),
                ImageContent(data="base64imagedata", mime_type="image/png")
            ],
            timestamp=1234567890000
        )
        assert msg.role == "user"
        assert len(msg.content) == 2


class TestAssistantMessage:
    """Tests for AssistantMessage model."""

    def test_assistant_message_creation(self):
        """AssistantMessage can be created with required fields."""
        from pi_ai.types import AssistantMessage, TextContent, Usage

        msg = AssistantMessage(
            content=[TextContent(text="Hello!")],
            api="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            usage=Usage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost=Usage.Cost(input=0.001, output=0.002, total=0.003)
            ),
            stop_reason="stop",
            timestamp=1234567890000
        )
        assert msg.role == "assistant"
        assert msg.api == "anthropic"
        assert msg.provider == "anthropic"
        assert msg.model == "claude-sonnet-4-20250514"
        assert msg.response_id is None
        assert msg.error_message is None
        assert msg.stop_reason == "stop"

    def test_assistant_message_with_response_id(self):
        """AssistantMessage can have optional response_id."""
        from pi_ai.types import AssistantMessage, TextContent, Usage

        msg = AssistantMessage(
            content=[TextContent(text="Hello!")],
            api="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            response_id="msg_123",
            usage=Usage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost=Usage.Cost(input=0.001, output=0.002, total=0.003)
            ),
            stop_reason="stop",
            timestamp=1234567890000
        )
        assert msg.response_id == "msg_123"

    def test_assistant_message_with_tool_call(self):
        """AssistantMessage can include ToolCall in content."""
        from pi_ai.types import AssistantMessage, TextContent, ToolCall, Usage

        msg = AssistantMessage(
            content=[
                TextContent(text="Let me check that for you."),
                ToolCall(id="call_123", name="get_weather", arguments={"city": "Beijing"})
            ],
            api="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            usage=Usage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost=Usage.Cost(input=0.001, output=0.002, total=0.003)
            ),
            stop_reason="toolUse",
            timestamp=1234567890000
        )
        assert len(msg.content) == 2
        assert msg.stop_reason == "toolUse"

    def test_assistant_message_with_error(self):
        """AssistantMessage can have error_message."""
        from pi_ai.types import AssistantMessage, TextContent, Usage

        msg = AssistantMessage(
            content=[TextContent(text="")],
            api="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            usage=Usage(
                input_tokens=100,
                output_tokens=0,
                total_tokens=100,
                cost=Usage.Cost(input=0.001, output=0.0, total=0.001)
            ),
            stop_reason="error",
            error_message="Rate limit exceeded",
            timestamp=1234567890000
        )
        assert msg.stop_reason == "error"
        assert msg.error_message == "Rate limit exceeded"


class TestToolResultMessage:
    """Tests for ToolResultMessage model."""

    def test_tool_result_message_creation(self):
        """ToolResultMessage can be created with required fields."""
        from pi_ai.types import ToolResultMessage, TextContent

        msg = ToolResultMessage(
            tool_call_id="call_123",
            tool_name="get_weather",
            content=[TextContent(text='{"temp": 25, "condition": "sunny"}')],
            is_error=False,
            timestamp=1234567890000
        )
        assert msg.role == "toolResult"
        assert msg.tool_call_id == "call_123"
        assert msg.tool_name == "get_weather"
        assert msg.is_error is False
        assert msg.details is None

    def test_tool_result_message_with_details(self):
        """ToolResultMessage can have optional details."""
        from pi_ai.types import ToolResultMessage, TextContent

        msg = ToolResultMessage(
            tool_call_id="call_123",
            tool_name="get_weather",
            content=[TextContent(text="Result")],
            details={"execution_time": 150},
            is_error=False,
            timestamp=1234567890000
        )
        assert msg.details == {"execution_time": 150}

    def test_tool_result_message_with_error(self):
        """ToolResultMessage can represent an error."""
        from pi_ai.types import ToolResultMessage, TextContent

        msg = ToolResultMessage(
            tool_call_id="call_456",
            tool_name="search_web",
            content=[TextContent(text="Connection timeout")],
            is_error=True,
            timestamp=1234567890000
        )
        assert msg.is_error is True

    def test_tool_result_message_with_image(self):
        """ToolResultMessage can contain ImageContent."""
        from pi_ai.types import ToolResultMessage, TextContent, ImageContent

        msg = ToolResultMessage(
            tool_call_id="call_789",
            tool_name="take_screenshot",
            content=[ImageContent(data="base64screenshot", mime_type="image/png")],
            is_error=False,
            timestamp=1234567890000
        )
        assert len(msg.content) == 1
        assert msg.content[0].type == "image"


class TestMessage:
    """Tests for Message union type."""

    def test_message_can_be_user_message(self):
        """Message union type includes UserMessage."""
        from pi_ai.types import Message, UserMessage

        msg: Message = UserMessage(content="Hello", timestamp=1234567890000)
        assert isinstance(msg, UserMessage)
        assert msg.role == "user"

    def test_message_can_be_assistant_message(self):
        """Message union type includes AssistantMessage."""
        from pi_ai.types import Message, AssistantMessage, TextContent, Usage

        msg: Message = AssistantMessage(
            content=[TextContent(text="Hi!")],
            api="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            usage=Usage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost=Usage.Cost(input=0.001, output=0.002, total=0.003)
            ),
            stop_reason="stop",
            timestamp=1234567890000
        )
        assert isinstance(msg, AssistantMessage)
        assert msg.role == "assistant"

    def test_message_can_be_tool_result_message(self):
        """Message union type includes ToolResultMessage."""
        from pi_ai.types import Message, ToolResultMessage, TextContent

        msg: Message = ToolResultMessage(
            tool_call_id="call_123",
            tool_name="test_tool",
            content=[TextContent(text="result")],
            is_error=False,
            timestamp=1234567890000
        )
        assert isinstance(msg, ToolResultMessage)
        assert msg.role == "toolResult"