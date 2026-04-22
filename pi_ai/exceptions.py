"""
pi-ai exceptions module.

Custom exception classes for pi-ai error handling.
"""


class PiAIError(Exception):
    """Base exception class for all pi-ai errors."""

    pass


class ProviderNotFoundError(PiAIError):
    """Raised when a requested provider is not found in the registry."""

    pass


class StreamError(PiAIError):
    """Raised when there's an error during streaming operations."""

    pass


class ToolValidationError(PiAIError):
    """Raised when tool parameter validation fails."""

    pass