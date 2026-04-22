"""
Tests for pi-ai validation utilities.

Tests the tool argument validation functions.
"""

import pytest
from pydantic import BaseModel, Field

from pi_ai.utils import (
    validate_tool_arguments,
    validate_tool_arguments_strict,
    get_validation_errors,
    is_valid_tool_arguments,
)
from pi_ai.exceptions import ToolValidationError


# -----------------------------------------------------------------------------
# Test Schemas
# -----------------------------------------------------------------------------

class WeatherArgs(BaseModel):
    """Sample schema for weather tool arguments."""
    location: str
    unit: str = Field(default="celsius")
    days: int = Field(default=1, ge=1, le=7)


class StrictArgs(BaseModel):
    """Sample schema for strict validation testing."""
    count: int
    name: str


class OptionalArgs(BaseModel):
    """Sample schema with optional fields."""
    required: str
    optional: str | None = None
    with_default: str = "default_value"


class NestedArgs(BaseModel):
    """Sample schema with nested structure."""
    outer: str
    inner: dict[str, int]


# -----------------------------------------------------------------------------
# validate_tool_arguments Tests
# -----------------------------------------------------------------------------

class TestValidateToolArguments:
    """Tests for validate_tool_arguments function."""

    def test_valid_arguments_with_defaults(self):
        """Test validation with valid arguments, defaults applied."""
        args = {"location": "San Francisco"}
        validated = validate_tool_arguments(WeatherArgs, args)

        assert validated["location"] == "San Francisco"
        assert validated["unit"] == "celsius"  # Default applied
        assert validated["days"] == 1  # Default applied

    def test_valid_arguments_all_fields(self):
        """Test validation with all fields provided."""
        args = {"location": "SF", "unit": "fahrenheit", "days": 3}
        validated = validate_tool_arguments(WeatherArgs, args)

        assert validated["location"] == "SF"
        assert validated["unit"] == "fahrenheit"
        assert validated["days"] == 3

    def test_type_coercion(self):
        """Test that type coercion works (string to int)."""
        args = {"location": "SF", "days": "5"}  # String "5"
        validated = validate_tool_arguments(WeatherArgs, args)

        assert validated["days"] == 5  # Coerced to int

    def test_missing_required_field_raises(self):
        """Test that missing required field raises ToolValidationError."""
        args = {"unit": "fahrenheit"}  # Missing 'location'

        with pytest.raises(ToolValidationError) as exc_info:
            validate_tool_arguments(WeatherArgs, args)

        assert "location" in str(exc_info.value)

    def test_invalid_constraint_raises(self):
        """Test that constraint violation raises ToolValidationError."""
        args = {"location": "SF", "days": 10}  # days > 7 (max is 7)

        with pytest.raises(ToolValidationError):
            validate_tool_arguments(WeatherArgs, args)

    def test_invalid_type_raises(self):
        """Test that invalid type raises ToolValidationError."""
        args = {"location": "SF", "days": "not_a_number"}

        with pytest.raises(ToolValidationError):
            validate_tool_arguments(WeatherArgs, args)

    def test_optional_fields(self):
        """Test validation with optional fields."""
        args = {"required": "value"}
        validated = validate_tool_arguments(OptionalArgs, args)

        assert validated["required"] == "value"
        assert validated["optional"] is None
        assert validated["with_default"] == "default_value"

    def test_optional_fields_provided(self):
        """Test validation when optional fields are provided."""
        args = {"required": "value", "optional": "opt", "with_default": "custom"}
        validated = validate_tool_arguments(OptionalArgs, args)

        assert validated["optional"] == "opt"
        assert validated["with_default"] == "custom"

    def test_nested_structure(self):
        """Test validation with nested dictionary."""
        args = {"outer": "test", "inner": {"a": 1, "b": 2}}
        validated = validate_tool_arguments(NestedArgs, args)

        assert validated["outer"] == "test"
        assert validated["inner"]["a"] == 1


# -----------------------------------------------------------------------------
# validate_tool_arguments_strict Tests
# -----------------------------------------------------------------------------

class TestValidateToolArgumentsStrict:
    """Tests for validate_tool_arguments_strict function."""

    def test_valid_strict_arguments(self):
        """Test strict validation with correct types."""
        args = {"count": 5, "name": "test"}
        validated = validate_tool_arguments_strict(StrictArgs, args)

        assert validated["count"] == 5
        assert validated["name"] == "test"

    def test_strict_rejects_type_coercion(self):
        """Test that strict mode rejects type coercion."""
        args = {"count": "5", "name": "test"}  # String instead of int

        with pytest.raises(ToolValidationError):
            validate_tool_arguments_strict(StrictArgs, args)

    def test_strict_missing_field_raises(self):
        """Test that strict mode raises for missing fields."""
        args = {"name": "test"}  # Missing 'count'

        with pytest.raises(ToolValidationError):
            validate_tool_arguments_strict(StrictArgs, args)


# -----------------------------------------------------------------------------
# get_validation_errors Tests
# -----------------------------------------------------------------------------

class TestGetValidationErrors:
    """Tests for get_validation_errors function."""

    def test_valid_returns_empty_list(self):
        """Test that valid arguments return empty list."""
        args = {"location": "SF"}
        errors = get_validation_errors(WeatherArgs, args)

        assert errors == []

    def test_invalid_returns_errors(self):
        """Test that invalid arguments return error list."""
        args = {"days": 100}  # Missing location, invalid days
        errors = get_validation_errors(WeatherArgs, args)

        assert len(errors) >= 1

    def test_error_structure(self):
        """Test structure of returned errors."""
        args = {}  # Missing all required
        errors = get_validation_errors(WeatherArgs, args)

        # Each error should have 'loc', 'msg', 'type'
        for error in errors:
            assert "loc" in error
            assert "msg" in error
            assert "type" in error


# -----------------------------------------------------------------------------
# is_valid_tool_arguments Tests
# -----------------------------------------------------------------------------

class TestIsValidToolArguments:
    """Tests for is_valid_tool_arguments function."""

    def test_valid_returns_true(self):
        """Test that valid arguments return True."""
        args = {"location": "SF"}
        assert is_valid_tool_arguments(WeatherArgs, args) == True

    def test_invalid_returns_false(self):
        """Test that invalid arguments return False."""
        args = {}  # Missing required field
        assert is_valid_tool_arguments(WeatherArgs, args) == False

    def test_type_coercion_valid(self):
        """Test that type coercion makes arguments valid."""
        args = {"location": "SF", "days": "5"}  # String coerced to int
        assert is_valid_tool_arguments(WeatherArgs, args) == True

    def test_extra_fields_valid(self):
        """Test that extra fields are still valid (Pydantic allows by default)."""
        args = {"location": "SF", "extra": "ignored"}
        assert is_valid_tool_arguments(WeatherArgs, args) == True


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------

class TestValidationEdgeCases:
    """Tests for edge cases in validation."""

    def test_empty_dict(self):
        """Test validation with empty dictionary."""
        # Should fail if required fields missing
        assert is_valid_tool_arguments(WeatherArgs, {}) == False

        # Should pass if no required fields
        class NoRequired(BaseModel):
            optional: str = "default"

        assert is_valid_tool_arguments(NoRequired, {}) == True

    def test_none_value_handling(self):
        """Test handling of None values."""
        args = {"required": None}  # None for required str field
        assert is_valid_tool_arguments(OptionalArgs, args) == False

    def test_complex_nested_validation(self):
        """Test validation with complex nested structure."""
        class Complex(BaseModel):
            items: list[dict[str, int]]

        args = {"items": [{"a": 1}, {"b": 2}]}
        assert is_valid_tool_arguments(Complex, args) == True

        args_invalid = {"items": [{"a": "not_int"}]}
        assert is_valid_tool_arguments(Complex, args_invalid) == False

    def test_list_validation(self):
        """Test validation with list fields."""
        class ListArgs(BaseModel):
            items: list[str]

        args = {"items": ["a", "b", "c"]}
        assert is_valid_tool_arguments(ListArgs, args) == True

        # Pydantic 2.x doesn't coerce ints to strings in lists by default
        args_invalid = {"items": [1, 2, 3]}  # Ints not coerced to strings
        assert is_valid_tool_arguments(ListArgs, args_invalid) == False