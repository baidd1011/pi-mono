# tests/test_exceptions.py
"""Tests for pi_ai exceptions."""

import pytest

from pi_ai.exceptions import PiAIError, ProviderNotFoundError, StreamError, ToolValidationError


class TestPiAIError:
    """Tests for PiAIError base exception."""

    def test_pi_ai_error_is_exception(self):
        """PiAIError is an Exception."""
        assert issubclass(PiAIError, Exception)

    def test_pi_ai_error_can_be_raised(self):
        """PiAIError can be raised and caught."""
        with pytest.raises(PiAIError) as exc_info:
            raise PiAIError("Something went wrong")
        assert str(exc_info.value) == "Something went wrong"

    def test_pi_ai_error_with_message(self):
        """PiAIError can have a custom message."""
        error = PiAIError("Custom error message")
        assert str(error) == "Custom error message"


class TestProviderNotFoundError:
    """Tests for ProviderNotFoundError."""

    def test_provider_not_found_error_is_pi_ai_error(self):
        """ProviderNotFoundError inherits from PiAIError."""
        assert issubclass(ProviderNotFoundError, PiAIError)

    def test_provider_not_found_error_can_be_raised(self):
        """ProviderNotFoundError can be raised and caught."""
        with pytest.raises(ProviderNotFoundError) as exc_info:
            raise ProviderNotFoundError("Provider 'openai' not found")
        assert str(exc_info.value) == "Provider 'openai' not found"

    def test_provider_not_found_error_caught_as_pi_ai_error(self):
        """ProviderNotFoundError can be caught as PiAIError."""
        with pytest.raises(PiAIError):
            raise ProviderNotFoundError("Provider not found")


class TestStreamError:
    """Tests for StreamError."""

    def test_stream_error_is_pi_ai_error(self):
        """StreamError inherits from PiAIError."""
        assert issubclass(StreamError, PiAIError)

    def test_stream_error_can_be_raised(self):
        """StreamError can be raised and caught."""
        with pytest.raises(StreamError) as exc_info:
            raise StreamError("Stream interrupted")
        assert str(exc_info.value) == "Stream interrupted"

    def test_stream_error_caught_as_pi_ai_error(self):
        """StreamError can be caught as PiAIError."""
        with pytest.raises(PiAIError):
            raise StreamError("Stream error")


class TestToolValidationError:
    """Tests for ToolValidationError."""

    def test_tool_validation_error_is_pi_ai_error(self):
        """ToolValidationError inherits from PiAIError."""
        assert issubclass(ToolValidationError, PiAIError)

    def test_tool_validation_error_can_be_raised(self):
        """ToolValidationError can be raised and caught."""
        with pytest.raises(ToolValidationError) as exc_info:
            raise ToolValidationError("Invalid tool parameters")
        assert str(exc_info.value) == "Invalid tool parameters"

    def test_tool_validation_error_caught_as_pi_ai_error(self):
        """ToolValidationError can be caught as PiAIError."""
        with pytest.raises(PiAIError):
            raise ToolValidationError("Validation failed")