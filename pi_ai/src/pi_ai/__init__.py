"""
pi-ai: Unified LLM API with streaming support for Bailian and OpenAI.

This package provides a unified interface for working with multiple LLM providers,
including Bailian (Alibaba/Qwen) and OpenAI, with full streaming support.
"""

# Core types
from pi_ai.types import (
    # Content types
    TextContent,
    ThinkingContent,
    ImageContent,
    ToolCall,
    ContentBlock,
    # Message types
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    Message,
    # Context and Model
    Context,
    Model,
    Tool,
    # Usage
    Usage,
    StopReason,
    ThinkingLevel,
    # Events
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
    AssistantMessageEvent,
)

# Stream module
from pi_ai.stream import (
    EventStream,
    AssistantMessageEventStream,
    stream_simple,
    complete_simple,
    StreamFunction,
    CompleteFunction,
)

# Registry
from pi_ai.registry import (
    ProviderInterface,
    ProviderProtocol,
    register_provider,
    get_provider,
    get_providers,
    clear_providers,
)

# Providers
from pi_ai.providers import (
    BailianProvider,
    OpenAIProvider,
    FauxProvider,
)

# Utilities
from pi_ai.utils import (
    validate_tool_arguments,
    validate_tool_arguments_strict,
    get_validation_errors,
    is_valid_tool_arguments,
)

# Exceptions
from pi_ai.exceptions import (
    PiAIError,
    ProviderNotFoundError,
    StreamError,
    ToolValidationError,
)

# Models
from pi_ai.models import (
    BAILIAN_MODELS,
    OPENAI_MODELS,
    ALL_MODELS,
    get_model,
    get_models,
)

# Re-export Pydantic for convenience
from pydantic import BaseModel, Field

__version__ = "0.1.0"

__all__ = [
    # Types
    "TextContent",
    "ThinkingContent",
    "ImageContent",
    "ToolCall",
    "ContentBlock",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "Message",
    "Context",
    "Model",
    "Tool",
    "Usage",
    "StopReason",
    "ThinkingLevel",
    "StartEvent",
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThinkingStartEvent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "DoneEvent",
    "ErrorEvent",
    "AssistantMessageEvent",
    # Stream
    "EventStream",
    "AssistantMessageEventStream",
    "stream_simple",
    "complete_simple",
    "StreamFunction",
    "CompleteFunction",
    # Registry
    "ProviderInterface",
    "ProviderProtocol",
    "register_provider",
    "get_provider",
    "get_providers",
    "clear_providers",
    # Providers
    "BailianProvider",
    "OpenAIProvider",
    "FauxProvider",
    # Utilities
    "validate_tool_arguments",
    "validate_tool_arguments_strict",
    "get_validation_errors",
    "is_valid_tool_arguments",
    # Exceptions
    "PiAIError",
    "ProviderNotFoundError",
    "StreamError",
    "ToolValidationError",
    # Models
    "BAILIAN_MODELS",
    "OPENAI_MODELS",
    "ALL_MODELS",
    "get_model",
    "get_models",
    # Pydantic
    "BaseModel",
    "Field",
]