# Agent types: AgentMessage, AgentTool, AgentEvent
"""
Tool types for pi-agent-core.

This module defines types for tool execution in the agent runtime:
- AgentToolResult: Result returned from tool execution
- AgentToolCall: Extracted tool call from assistant message
- AgentTool: Tool definition with execute function
- ToolExecutionMode: Sequential or parallel execution
- AgentToolUpdateCallback: Callback for streaming partial results
"""

from typing import Any, Callable, Awaitable, Optional, Union, Literal, List
from dataclasses import dataclass

from pi_ai import TextContent, ImageContent


# -----------------------------------------------------------------------------
# Tool Execution Mode
# -----------------------------------------------------------------------------

ToolExecutionMode = Literal["sequential", "parallel"]


# -----------------------------------------------------------------------------
# Agent Tool Result
# -----------------------------------------------------------------------------

@dataclass
class AgentToolResult[TDetails]:
    """Result returned from tool execution."""
    content: List[Union[TextContent, ImageContent]]
    details: TDetails


# -----------------------------------------------------------------------------
# Agent Tool Update Callback
# -----------------------------------------------------------------------------

AgentToolUpdateCallback = Callable[[AgentToolResult], None]


# -----------------------------------------------------------------------------
# Agent Tool Call
# -----------------------------------------------------------------------------

@dataclass
class AgentToolCall:
    """Tool call content block from assistant message."""
    id: str
    name: str
    arguments: dict[str, Any]


# -----------------------------------------------------------------------------
# Agent Tool
# -----------------------------------------------------------------------------

@dataclass
class AgentTool[TParameters, TDetails]:
    """Tool definition for agent runtime."""
    name: str
    label: str  # Human-readable label for UI
    description: str
    parameters: Any  # Schema (Pydantic model or dict)

    # Execute the tool - throw on failure
    execute: Callable[
        [str, TParameters, Optional[Any], Optional[AgentToolUpdateCallback]],
        Awaitable[AgentToolResult[TDetails]]
    ]

    # Optional compatibility shim for raw tool-call arguments
    prepare_arguments: Optional[Callable[[Any], TParameters]] = None

    # Per-tool execution mode override
    execution_mode: Optional[ToolExecutionMode] = None