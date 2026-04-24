# Agent types: AgentMessage, AgentTool, AgentEvent
"""
Types for pi-agent-core.

This module defines types for the agent runtime:
- AgentToolResult: Result returned from tool execution
- AgentToolCall: Extracted tool call from assistant message
- AgentTool: Tool definition with execute function
- ToolExecutionMode: Sequential or parallel execution
- AgentToolUpdateCallback: Callback for streaming partial results
- ThinkingLevel: Levels for reasoning/thinking capabilities
- AgentState: Read-only snapshot of agent's current state
- AgentContext: Context passed into agent loop for LLM calls
- AgentLoopConfig: Configuration for agent loop execution
"""

from typing import Any, Callable, Awaitable, Optional, Union, Literal, List, FrozenSet
from typing import TypedDict
from dataclasses import dataclass, field

from pi_ai import TextContent, ImageContent, Model, AssistantMessageEventStream, Context


# -----------------------------------------------------------------------------
# Thinking Level
# -----------------------------------------------------------------------------

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


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


# -----------------------------------------------------------------------------
# Agent State
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentState:
    """Public agent state (read-only snapshot)."""
    system_prompt: str
    model: Model
    thinking_level: ThinkingLevel

    # Tools and messages (copied on access)
    tools: List[Any]  # List[AgentTool]
    messages: List[Any]  # List[AgentMessage]

    # Runtime state (read-only)
    is_streaming: bool
    streaming_message: Optional[Any]
    pending_tool_calls: FrozenSet[str]
    error_message: Optional[str]


# -----------------------------------------------------------------------------
# Agent Context
# -----------------------------------------------------------------------------

@dataclass
class AgentContext:
    """Context snapshot passed into agent loop."""
    system_prompt: str
    messages: List[Any]  # List[AgentMessage]
    tools: Optional[List[Any]] = None  # List[AgentTool]


# -----------------------------------------------------------------------------
# Agent Loop Configuration Types
# -----------------------------------------------------------------------------

@dataclass
class BeforeToolCallResult:
    """Result returned from before_tool_call hook."""
    block: bool = False
    reason: Optional[str] = None


@dataclass
class AfterToolCallResult:
    """Partial override returned from after_tool_call hook."""
    content: Optional[List[Union[TextContent, ImageContent]]] = None
    details: Optional[Any] = None
    is_error: Optional[bool] = None


@dataclass
class BeforeToolCallContext:
    """Context passed to before_tool_call hook."""
    assistant_message: Any  # AssistantMessage
    tool_call: AgentToolCall
    args: Any  # Validated arguments
    context: AgentContext


@dataclass
class AfterToolCallContext:
    """Context passed to after_tool_call hook."""
    assistant_message: Any  # AssistantMessage
    tool_call: AgentToolCall
    args: Any  # Validated arguments
    result: AgentToolResult
    is_error: bool
    context: AgentContext


# Stream function type
StreamFn = Callable[[Context, Model], Awaitable[AssistantMessageEventStream]]


@dataclass
class AgentLoopConfig:
    """Configuration for agent loop execution."""
    model: Model

    # Message transformation - must not throw/reject
    convert_to_llm: Callable[[List[Any]], Union[List[Any], Awaitable[List[Any]]]]

    # Optional context transform before convert_to_llm
    transform_context: Optional[Callable[[List[Any], Optional[Any]], Awaitable[List[Any]]]] = None

    # API key resolution for expiring tokens
    get_api_key: Optional[Callable[[str], Union[str, Awaitable[str]]]] = None

    # Message queues
    get_steering_messages: Optional[Callable[[], Awaitable[List[Any]]]] = None
    get_follow_up_messages: Optional[Callable[[], Awaitable[List[Any]]]] = None

    # Tool execution mode
    tool_execution: ToolExecutionMode = "parallel"

    # Tool execution hooks
    before_tool_call: Optional[Callable[[BeforeToolCallContext, Optional[Any]], Awaitable[Optional[BeforeToolCallResult]]]] = None
    after_tool_call: Optional[Callable[[AfterToolCallContext, Optional[Any]], Awaitable[Optional[AfterToolCallResult]]]] = None

    # Additional stream options
    api_key: Optional[str] = None
    thinking_level: Optional[ThinkingLevel] = None


# -----------------------------------------------------------------------------
# Agent Event Types
# -----------------------------------------------------------------------------

class AgentStartEvent(TypedDict):
    """Event emitted when agent starts running."""
    type: Literal["agent_start"]


class AgentEndEvent(TypedDict):
    """Event emitted when agent stops running."""
    type: Literal["agent_end"]
    messages: List[Any]  # List[AgentMessage]


class TurnStartEvent(TypedDict):
    """Event emitted at the start of each turn."""
    type: Literal["turn_start"]


class TurnEndEvent(TypedDict):
    """Event emitted at the end of each turn."""
    type: Literal["turn_end"]
    message: Any  # AgentMessage
    tool_results: List[Any]  # List[ToolResultMessage]


class MessageStartEvent(TypedDict):
    """Event emitted when a new message starts streaming."""
    type: Literal["message_start"]
    message: Any  # AgentMessage


class MessageUpdateEvent(TypedDict):
    """Event emitted during message streaming."""
    type: Literal["message_update"]
    message: Any  # AgentMessage
    assistant_message_event: Any  # AssistantMessageEvent


class MessageEndEvent(TypedDict):
    """Event emitted when message streaming completes."""
    type: Literal["message_end"]
    message: Any  # AgentMessage


class ToolExecutionStartEvent(TypedDict):
    """Event emitted when tool execution starts."""
    type: Literal["tool_execution_start"]
    tool_call_id: str
    tool_name: str
    args: Any


class ToolExecutionUpdateEvent(TypedDict):
    """Event emitted during tool execution with partial results."""
    type: Literal["tool_execution_update"]
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any


class ToolExecutionEndEvent(TypedDict):
    """Event emitted when tool execution completes."""
    type: Literal["tool_execution_end"]
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


# Union of all agent events
AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
]

# Event sink callback type
AgentEventSink = Callable[[AgentEvent], None]