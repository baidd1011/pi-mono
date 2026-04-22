"""
Tests for pi-ai providers.

Tests the FauxProvider, BailianProvider, and OpenAIProvider implementations.
"""

import asyncio
import pytest

from pi_ai.providers import (
    FauxProvider,
    BailianProvider,
    OpenAIProvider,
)
from pi_ai.types import (
    Context,
    Model,
    UserMessage,
    AssistantMessage,
    Usage,
    TextContent,
    ToolResultMessage,
)
from pi_ai.stream import AssistantMessageEventStream
from pi_ai.models import get_model


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_model() -> Model:
    """Sample model configuration for testing."""
    return Model(
        id="test-model",
        name="Test Model",
        api="test",
        provider="test",
        base_url="https://test.example.com/v1",
        reasoning=False,
        input=["text"],
        cost=Usage.Cost(input=0.01, output=0.02, total=0.03),
        context_window=4096,
        max_tokens=1024,
    )


@pytest.fixture
def sample_context() -> Context:
    """Sample conversation context for testing."""
    return Context(
        system_prompt="You are a helpful assistant.",
        messages=[
            UserMessage(
                role="user",
                content="Hello, world!",
                timestamp=1000,
            )
        ],
    )


# -----------------------------------------------------------------------------
# FauxProvider Tests
# -----------------------------------------------------------------------------

class TestFauxProvider:
    """Tests for the FauxProvider mock provider."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self, sample_model: Model, sample_context: Context):
        """Test that FauxProvider returns a simple text response."""
        provider = FauxProvider(responses=[[("text", "Hello!")]])

        stream = await provider.stream(sample_context, sample_model)
        result = await stream.result()

        assert result.role == "assistant"
        assert len(result.content) >= 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "Hello!"

    @pytest.mark.asyncio
    async def test_thinking_then_text_response(self, sample_model: Model, sample_context: Context):
        """Test that FauxProvider returns thinking followed by text."""
        provider = FauxProvider(
            responses=[[("thinking", "Let me think..."), ("text", "The answer is 42.")]]
        )

        stream = await provider.stream(sample_context, sample_model)
        result = await stream.result()

        # Should have thinking and text content
        has_thinking = any(
            hasattr(c, 'thinking') and c.thinking for c in result.content
        )
        has_text = any(
            hasattr(c, 'text') and "42" in c.text for c in result.content
        )
        assert has_thinking or has_text

    @pytest.mark.asyncio
    async def test_event_iteration(self, sample_model: Model, sample_context: Context):
        """Test that events can be iterated from the stream."""
        provider = FauxProvider(responses=[[("text", "Test response")]])

        stream = await provider.stream(sample_context, sample_model)

        events = []
        async for event in stream:
            events.append(event)

        # Should have start, textStart, textDelta(s), textEnd, done events
        assert len(events) >= 5
        assert events[0]["type"] == "start"
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_error_response(self, sample_model: Model, sample_context: Context):
        """Test that FauxProvider can return an error."""
        provider = FauxProvider(error="Simulated error")

        stream = await provider.stream(sample_context, sample_model)
        result = await stream.result()

        assert result.stop_reason == "error"
        assert result.error_message == "Simulated error"

    @pytest.mark.asyncio
    async def test_multiple_responses_cycle(self, sample_model: Model, sample_context: Context):
        """Test that FauxProvider cycles through multiple responses."""
        provider = FauxProvider(
            responses=[[("text", "First")], [("text", "Second")], [("text", "Third")]]
        )

        # First call
        result1 = await (await provider.stream(sample_context, sample_model)).result()
        assert "First" in str(result1.content)

        # Second call
        result2 = await (await provider.stream(sample_context, sample_model)).result()
        assert "Second" in str(result2.content)

        # Third call
        result3 = await (await provider.stream(sample_context, sample_model)).result()
        assert "Third" in str(result3.content)

        # Fourth call - should cycle back to first
        result4 = await (await provider.stream(sample_context, sample_model)).result()
        assert "First" in str(result4.content)

    @pytest.mark.asyncio
    async def test_stream_simple_same_as_stream(self, sample_model: Model, sample_context: Context):
        """Test that stream_simple behaves like stream."""
        provider = FauxProvider(responses=[[("text", "Hello!")]])

        stream1 = await provider.stream(sample_context, sample_model)
        stream2 = await provider.stream_simple(sample_context, sample_model)

        result1 = await stream1.result()
        result2 = await stream2.result()

        assert result1.content[0].text == result2.content[0].text

    @pytest.mark.asyncio
    async def test_delay_between_events(self, sample_model: Model, sample_context: Context):
        """Test that delay works between events."""
        provider = FauxProvider(responses=[[("text", "Delayed")]], delay=0.01)

        start_time = asyncio.get_event_loop().time()
        stream = await provider.stream(sample_context, sample_model)

        # Iterate through events
        async for event in stream:
            pass

        elapsed = asyncio.get_event_loop().time() - start_time
        # Should have some delay from multiple events
        assert elapsed >= 0.01

    def test_reset_call_count(self, sample_model: Model, sample_context: Context):
        """Test that reset() resets the call counter."""
        provider = FauxProvider(responses=[[("text", "A")], [("text", "B")]])

        assert provider._call_count == 0
        provider._call_count = 5
        provider.reset()
        assert provider._call_count == 0


# -----------------------------------------------------------------------------
# FauxProvider Factory Tests
# -----------------------------------------------------------------------------

class TestFauxProviderFactories:
    """Tests for FauxProvider factory functions."""

    @pytest.mark.asyncio
    async def test_create_text_faux_provider(self, sample_model: Model, sample_context: Context):
        """Test create_text_faux_provider factory."""
        from pi_ai.providers.faux import create_text_faux_provider

        provider = create_text_faux_provider("Custom text")
        result = await (await provider.stream(sample_context, sample_model)).result()
        assert "Custom text" in str(result.content)

    @pytest.mark.asyncio
    async def test_create_thinking_faux_provider(self, sample_model: Model, sample_context: Context):
        """Test create_thinking_faux_provider factory."""
        from pi_ai.providers.faux import create_thinking_faux_provider

        provider = create_thinking_faux_provider("Thinking...", "Answer")
        result = await (await provider.stream(sample_context, sample_model)).result()
        assert len(result.content) >= 1

    @pytest.mark.asyncio
    async def test_create_error_faux_provider(self, sample_model: Model, sample_context: Context):
        """Test create_error_faux_provider factory."""
        from pi_ai.providers.faux import create_error_faux_provider

        provider = create_error_faux_provider("Test error")
        result = await (await provider.stream(sample_context, sample_model)).result()
        assert result.stop_reason == "error"
        assert result.error_message == "Test error"


# -----------------------------------------------------------------------------
# BailianProvider Tests
# -----------------------------------------------------------------------------

class TestBailianProvider:
    """Tests for the Bailian provider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = BailianProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert provider.base_url == BailianProvider.BASE_URL

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        provider = BailianProvider(api_key="test_key", base_url="https://custom.url")
        assert provider.base_url == "https://custom.url"

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="API key required"):
            BailianProvider()

    def test_get_headers(self):
        """Test that headers are correctly formatted."""
        provider = BailianProvider(api_key="test_key")
        headers = provider._get_headers()
        assert headers["Authorization"] == "Bearer test_key"
        assert headers["Content-Type"] == "application/json"

    def test_convert_user_message_string(self):
        """Test converting a user message with string content."""
        provider = BailianProvider(api_key="test_key")
        message = UserMessage(role="user", content="Hello", timestamp=1000)
        converted = provider._convert_message(message)

        assert converted["role"] == "user"
        assert converted["content"] == "Hello"

    def test_convert_user_message_list(self):
        """Test converting a user message with list content."""
        provider = BailianProvider(api_key="test_key")
        message = UserMessage(
            role="user",
            content=[TextContent(text="Hello")],
            timestamp=1000
        )
        converted = provider._convert_message(message)

        assert converted["role"] == "user"
        assert isinstance(converted["content"], list)
        assert converted["content"][0]["type"] == "text"

    def test_build_request_body_basic(self, sample_model: Model, sample_context: Context):
        """Test building basic request body."""
        provider = BailianProvider(api_key="test_key")
        body = provider._build_request_body(sample_context, sample_model)

        assert body["model"] == sample_model.id
        assert body["stream"] == True
        assert "messages" in body

    def test_build_request_body_with_thinking(self, sample_model: Model, sample_context: Context):
        """Test building request body with enable_thinking."""
        provider = BailianProvider(api_key="test_key")
        body = provider._build_request_body(
            sample_context, sample_model, enable_thinking=True
        )

        assert body["enable_thinking"] == True


# -----------------------------------------------------------------------------
# OpenAIProvider Tests
# -----------------------------------------------------------------------------

class TestOpenAIProvider:
    """Tests for the OpenAI provider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = OpenAIProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert provider.base_url == OpenAIProvider.BASE_URL

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        provider = OpenAIProvider(api_key="test_key", base_url="https://custom.url")
        assert provider.base_url == "https://custom.url"

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="API key required"):
            OpenAIProvider()

    def test_client_property(self):
        """Test that client property creates AsyncOpenAI client."""
        provider = OpenAIProvider(api_key="test_key")
        client = provider.client
        assert client is not None
        # Client should be cached
        client2 = provider.client
        assert client is client2

    def test_convert_user_message_string(self):
        """Test converting a user message with string content."""
        provider = OpenAIProvider(api_key="test_key")
        message = UserMessage(role="user", content="Hello", timestamp=1000)
        converted = provider._convert_message(message)

        assert converted["role"] == "user"
        assert converted["content"] == "Hello"

    def test_convert_user_message_list(self):
        """Test converting a user message with list content."""
        provider = OpenAIProvider(api_key="test_key")
        message = UserMessage(
            role="user",
            content=[TextContent(text="Hello")],
            timestamp=1000
        )
        converted = provider._convert_message(message)

        assert converted["role"] == "user"
        assert isinstance(converted["content"], list)


# -----------------------------------------------------------------------------
# Provider Registry Tests
# -----------------------------------------------------------------------------

class TestProviderRegistry:
    """Tests for provider registration."""

    def test_register_and_get_provider(self):
        """Test registering and retrieving a provider."""
        from pi_ai.registry import register_provider, get_provider, clear_providers

        clear_providers()
        provider = FauxProvider(responses=[[("text", "Test")]])
        register_provider("faux", provider)

        retrieved = get_provider("faux")
        assert retrieved is provider

    def test_get_provider_not_found_raises(self):
        """Test that get_provider raises for unknown provider."""
        from pi_ai.registry import get_provider, clear_providers

        clear_providers()
        with pytest.raises(Exception):  # ProviderNotFoundError
            get_provider("nonexistent")

    def test_get_providers(self):
        """Test listing registered providers."""
        from pi_ai.registry import register_provider, get_providers, clear_providers

        clear_providers()
        register_provider("faux1", FauxProvider())
        register_provider("faux2", FauxProvider())

        providers = get_providers()
        assert "faux1" in providers
        assert "faux2" in providers

    def test_clear_providers(self):
        """Test clearing all registered providers."""
        from pi_ai.registry import register_provider, get_providers, clear_providers

        register_provider("faux", FauxProvider())
        assert len(get_providers()) >= 1

        clear_providers()
        assert len(get_providers()) == 0