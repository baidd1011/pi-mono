"""
pi-agent-core: Agent runtime with state management, tool execution, and event system.
"""

# Tool types
from pi_agent_core.types import (
    AgentTool,
    AgentToolResult,
    AgentToolCall,
    ToolExecutionMode,
    AgentToolUpdateCallback,
    AgentState,
    AgentContext,
    ThinkingLevel,
    AgentLoopConfig,
    BeforeToolCallResult,
    AfterToolCallResult,
    BeforeToolCallContext,
    AfterToolCallContext,
    StreamFn,
    AgentEvent,
    AgentEventSink,
)

# Agent class
from pi_agent_core.agent import (
    Agent,
    AgentOptions,
    DEFAULT_MODEL,
)

# Agent loop functions
from pi_agent_core.agent_loop import (
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
    AgentEventStream,
)

__version__ = "0.1.0"

__all__ = [
    # Tool types
    "AgentTool",
    "AgentToolResult",
    "AgentToolCall",
    "ToolExecutionMode",
    "AgentToolUpdateCallback",
    # State types
    "AgentState",
    "AgentContext",
    "ThinkingLevel",
    "AgentLoopConfig",
    "BeforeToolCallResult",
    "AfterToolCallResult",
    "BeforeToolCallContext",
    "AfterToolCallContext",
    "StreamFn",
    "AgentEvent",
    "AgentEventSink",
    # Agent class
    "Agent",
    "AgentOptions",
    "DEFAULT_MODEL",
    # Agent loop
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
    "AgentEventStream",
]