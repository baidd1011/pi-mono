"""
Tool validation utilities.

Provides validation functions for tool arguments against Pydantic schemas.
"""

from typing import Any, Type

from pydantic import BaseModel, ValidationError

from pi_ai.exceptions import ToolValidationError


def validate_tool_arguments(
    schema: Type[BaseModel],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate arguments against a Pydantic schema.

    Args:
        schema: The Pydantic model class defining the expected schema.
        arguments: The arguments dictionary to validate.

    Returns:
        The validated and potentially coerced arguments.

    Raises:
        ToolValidationError: If validation fails.

    Example:
        class WeatherArgs(BaseModel):
            location: str
            unit: str = "celsius"

        validated = validate_tool_arguments(WeatherArgs, {"location": "SF"})
        # Returns: {"location": "SF", "unit": "celsius"}
    """
    try:
        # Create and validate the model instance
        validated = schema.model_validate(arguments)
        # Return the validated data with defaults applied
        return validated.model_dump()
    except ValidationError as e:
        # Extract error details for a more helpful message
        errors = []
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  - {loc}: {msg}")

        error_msg = f"Tool argument validation failed:\n" + "\n".join(errors)
        raise ToolValidationError(error_msg) from e


def validate_tool_arguments_strict(
    schema: Type[BaseModel],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate arguments against a Pydantic schema in strict mode.

    Strict mode requires exact type matching and does not perform
    type coercion (e.g., string "123" won't be converted to int 123).

    Args:
        schema: The Pydantic model class defining the expected schema.
        arguments: The arguments dictionary to validate.

    Returns:
        The validated arguments.

    Raises:
        ToolValidationError: If validation fails.
    """
    try:
        validated = schema.model_validate(arguments, strict=True)
        return validated.model_dump()
    except ValidationError as e:
        errors = []
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  - {loc}: {msg}")

        error_msg = f"Tool argument validation failed (strict mode):\n" + "\n".join(errors)
        raise ToolValidationError(error_msg) from e


def get_validation_errors(
    schema: Type[BaseModel],
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Get validation errors without raising an exception.

    Args:
        schema: The Pydantic model class defining the expected schema.
        arguments: The arguments dictionary to validate.

    Returns:
        A list of error dictionaries, or an empty list if valid.

    Example:
        errors = get_validation_errors(MySchema, {"invalid": "data"})
        if errors:
            print("Validation failed:", errors)
    """
    try:
        schema.model_validate(arguments)
        return []
    except ValidationError as e:
        return e.errors()


def is_valid_tool_arguments(
    schema: Type[BaseModel],
    arguments: dict[str, Any],
) -> bool:
    """
    Check if arguments are valid against a Pydantic schema.

    Args:
        schema: The Pydantic model class defining the expected schema.
        arguments: The arguments dictionary to validate.

    Returns:
        True if valid, False otherwise.
    """
    try:
        schema.model_validate(arguments)
        return True
    except ValidationError:
        return False