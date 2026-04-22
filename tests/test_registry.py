# tests/test_registry.py
"""Tests for pi_ai registry module."""

import pytest

from pi_ai.registry import (
    ProviderInterface,
    register_provider,
    get_provider,
    get_providers,
    clear_providers,
)
from pi_ai.exceptions import ProviderNotFoundError


class MockProvider(ProviderInterface):
    """Mock provider for testing."""

    def __init__(self, name: str):
        self.name = name

    async def stream(self, context, model):
        """Mock stream implementation."""
        from pi_ai.stream import AssistantMessageEventStream
        return AssistantMessageEventStream()

    async def stream_simple(self, context, model):
        """Mock stream_simple implementation."""
        from pi_ai.stream import AssistantMessageEventStream
        return AssistantMessageEventStream()


class TestProviderInterface:
    """Tests for ProviderInterface ABC."""

    def test_provider_interface_is_abstract(self):
        """ProviderInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ProviderInterface()

    def test_provider_interface_requires_stream_method(self):
        """ProviderInterface subclasses must implement stream()."""

        class IncompleteProvider(ProviderInterface):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_provider_interface_subclass_can_be_created(self):
        """Complete ProviderInterface subclasses can be created."""
        provider = MockProvider("test")
        assert isinstance(provider, ProviderInterface)
        assert provider.name == "test"


class TestRegisterProvider:
    """Tests for register_provider function."""

    def setup_method(self):
        """Clear providers before each test."""
        clear_providers()

    def test_register_provider(self):
        """register_provider adds provider to registry."""
        provider = MockProvider("openai")
        register_provider("openai", provider)

        assert "openai" in get_providers()

    def test_register_multiple_providers(self):
        """register_provider can register multiple providers."""
        provider1 = MockProvider("openai")
        provider2 = MockProvider("anthropic")

        register_provider("openai", provider1)
        register_provider("anthropic", provider2)

        providers = get_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert len(providers) == 2

    def test_register_provider_replaces_existing(self):
        """register_provider replaces existing provider."""
        provider1 = MockProvider("old")
        provider2 = MockProvider("new")

        register_provider("test", provider1)
        register_provider("test", provider2)

        retrieved = get_provider("test")
        assert retrieved.name == "new"


class TestGetProvider:
    """Tests for get_provider function."""

    def setup_method(self):
        """Clear providers before each test."""
        clear_providers()

    def test_get_provider_returns_registered_provider(self):
        """get_provider returns the registered provider."""
        provider = MockProvider("test")
        register_provider("test", provider)

        retrieved = get_provider("test")
        assert retrieved is provider

    def test_get_provider_raises_for_unknown_provider(self):
        """get_provider raises ProviderNotFoundError for unknown provider."""
        with pytest.raises(ProviderNotFoundError, match="Provider 'unknown' not found"):
            get_provider("unknown")

    def test_get_provider_error_message_includes_available_providers(self):
        """get_provider error includes available providers."""
        register_provider("openai", MockProvider("openai"))
        register_provider("anthropic", MockProvider("anthropic"))

        try:
            get_provider("unknown")
        except ProviderNotFoundError as e:
            assert "openai" in str(e)
            assert "anthropic" in str(e)


class TestGetProviders:
    """Tests for get_providers function."""

    def setup_method(self):
        """Clear providers before each test."""
        clear_providers()

    def test_get_providers_returns_empty_list_when_empty(self):
        """get_providers returns empty list when no providers registered."""
        assert get_providers() == []

    def test_get_providers_returns_provider_names(self):
        """get_providers returns list of registered provider names."""
        register_provider("openai", MockProvider("openai"))
        register_provider("anthropic", MockProvider("anthropic"))

        providers = get_providers()
        assert set(providers) == {"openai", "anthropic"}

    def test_get_providers_returns_copy(self):
        """get_providers returns a copy that can be modified."""
        register_provider("test", MockProvider("test"))
        providers = get_providers()
        providers.append("fake")

        # Original registry should not be affected
        assert get_providers() == ["test"]


class TestClearProviders:
    """Tests for clear_providers function."""

    def test_clear_providers_removes_all(self):
        """clear_providers removes all registered providers."""
        register_provider("openai", MockProvider("openai"))
        register_provider("anthropic", MockProvider("anthropic"))

        clear_providers()

        assert get_providers() == []

    def test_clear_providers_on_empty_registry(self):
        """clear_providers works on empty registry."""
        clear_providers()  # Should not raise
        assert get_providers() == []