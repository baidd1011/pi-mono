"""
pi-ai: Unified LLM API with streaming support for Bailian and OpenAI.
"""

# Core types - will be added in subsequent tasks
# from .types import ...

# Stream API - will be added in subsequent tasks
# from .stream import ...

# Provider registry - will be added in subsequent tasks
# from .registry import ...

# Re-export Pydantic for schema definitions
from pydantic import BaseModel, Field

__version__ = "0.1.0"
__all__ = ["BaseModel", "Field"]