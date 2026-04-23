"""
pi-ai provider registry.

Provider registration and lookup for LLM API implementations.
"""

from abc import ABC, abstractmethod
from typing import Protocol

from pi_ai.exceptions import ProviderNotFoundError
from pi_ai.types import Context, Model, AssistantMessage
from pi_ai.stream import AssistantMessageEventStream


class ProviderInterface(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement the stream() and stream_simple() methods
    to provide streaming responses from their respective APIs.
    """

    @abstractmethod
    async def stream(self, context: Context, model: Model) -> AssistantMessageEventStream:
        """
        Stream assistant message events for a given context and model.

        Args:
            context: The conversation context including messages and tools.
            model: The model configuration to use.

        Returns:
            An AssistantMessageEventStream for iterating over response events.
        """
        pass

    @abstractmethod
    async def stream_simple(self, context: Context, model: Model) -> AssistantMessageEventStream:
        """
        Simple streaming wrapper (same as stream for most providers).

        Args:
            context: The conversation context including messages and tools.
            model: The model configuration to use.

        Returns:
            An AssistantMessageEventStream for iterating over response events.
        """
        pass


class ProviderProtocol(Protocol):
    """
    Protocol for providers that can be registered.

    Allows duck-typed providers that implement the required methods.
    """

    async def stream(self, context: Context, model: Model) -> AssistantMessageEventStream:
        ...

    async def stream_simple(self, context: Context, model: Model) -> AssistantMessageEventStream:
        ...


# Global provider registry
_providers: dict[str, ProviderInterface] = {}


def register_provider(name: str, provider: ProviderInterface) -> None:
    """
    Register a provider with the registry.

    Args:
        name: The provider name (e.g., "anthropic", "openai").
        provider: The provider instance implementing ProviderInterface.
    """
    _providers[name] = provider


def get_provider(name: str) -> ProviderInterface:
    """
    Get a provider from the registry.

    Args:
        name: The provider name to look up.

    Returns:
        The provider instance.

    Raises:
        ProviderNotFoundError: If the provider is not registered.
    """
    if name not in _providers:
        raise ProviderNotFoundError(f"Provider '{name}' not found. Available providers: {list(_providers.keys())}")
    return _providers[name]


def get_providers() -> list[str]:
    """
    Get all registered provider names.

    Returns:
        List of provider names.
    """
    return list(_providers.keys())


def clear_providers() -> None:
    """
    Clear all registered providers.

    Useful for testing and resetting the registry.
    """
    _providers.clear()