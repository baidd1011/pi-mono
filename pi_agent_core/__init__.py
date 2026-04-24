"""
pi-agent-core: Agent runtime with state management, tool execution, and event system.
"""

from .agent import Agent, AgentOptions, DEFAULT_MODEL, AgentMessage
from .agent_loop import (
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
)
from ._event_stream import AgentEventStream, create_agent_stream
from ._abort import AbortController, AbortSignal
from .types import (
    AgentState,
    AgentContext,
    AgentTool,
    AgentToolResult,
    AgentToolCall,
    AgentEvent,
    ToolExecutionMode,
    ThinkingLevel,
    AgentEventSink,
    StreamFn,
    AgentLoopConfig,
    BeforeToolCallContext,
    BeforeToolCallResult,
    AfterToolCallContext,
    AfterToolCallResult,
)

__version__ = "0.1.0"

__all__ = [
    # Agent
    "Agent",
    "AgentOptions",
    "DEFAULT_MODEL",
    "AgentMessage",
    # Loop
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
    "AgentLoopConfig",
    # Stream
    "AgentEventStream",
    "create_agent_stream",
    # Abort
    "AbortController",
    "AbortSignal",
    # Types
    "AgentState",
    "AgentContext",
    "AgentTool",
    "AgentToolResult",
    "AgentToolCall",
    "AgentEvent",
    "ToolExecutionMode",
    "ThinkingLevel",
    "AgentEventSink",
    "StreamFn",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "AfterToolCallContext",
    "AfterToolCallResult",
]