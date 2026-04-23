"""Agent class - stateful wrapper around agent loop."""

import asyncio
from typing import Callable, Optional, List, Union, Awaitable, Any, Literal
from dataclasses import dataclass, field

from pi_ai import (
    Model, Context, Message, Usage,
    AssistantMessageEventStream, ImageContent, TextContent,
    stream_simple,
)
from ._abort import AbortController, AbortSignal
from .types import (
    AgentState, AgentContext, AgentTool, AgentToolResult,
    AgentEvent, ToolExecutionMode, ThinkingLevel,
)

# Alias Message to AgentMessage for clarity
AgentMessage = Message


# Default model placeholder
DEFAULT_MODEL = Model(
    id="unknown", name="unknown", api="unknown", provider="unknown",
    base_url="", reasoning=False, input=["text"],
    cost=Usage.Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    context_window=0, max_tokens=0,
)


@dataclass
class _MutableAgentState:
    """Internal mutable state."""
    system_prompt: str = ""
    model: Model = field(default_factory=lambda: DEFAULT_MODEL)
    thinking_level: ThinkingLevel = "off"
    _tools: List[AgentTool] = field(default_factory=list)
    _messages: List[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    streaming_message: Optional[AgentMessage] = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error_message: Optional[str] = None

    @property
    def tools(self) -> List[AgentTool]:
        return self._tools.copy()

    @tools.setter
    def tools(self, value: List[AgentTool]):
        self._tools = value.copy()

    @property
    def messages(self) -> List[AgentMessage]:
        return self._messages.copy()

    @messages.setter
    def messages(self, value: List[AgentMessage]):
        self._messages = value.copy()


@dataclass
class AgentOptions:
    """Options for constructing an Agent."""
    initial_state: Optional[dict] = None
    convert_to_llm: Optional[Callable] = None
    transform_context: Optional[Callable] = None
    stream_fn: Optional[Callable] = None
    get_api_key: Optional[Callable] = None
    on_payload: Optional[Callable] = None
    on_response: Optional[Callable] = None
    before_tool_call: Optional[Callable] = None
    after_tool_call: Optional[Callable] = None
    steering_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    follow_up_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    session_id: Optional[str] = None
    thinking_budgets: Optional[dict] = None
    transport: Literal["sse", "websocket", "auto"] = "sse"
    max_retry_delay_ms: Optional[int] = None
    tool_execution: ToolExecutionMode = "parallel"


class Agent:
    """Stateful wrapper around the agent loop."""

    def __init__(self, options: Optional[AgentOptions] = None):
        options = options or AgentOptions()

        self._state = _MutableAgentState()
        if options.initial_state:
            if "system_prompt" in options.initial_state:
                self._state.system_prompt = options.initial_state["system_prompt"]
            if "model" in options.initial_state:
                self._state.model = options.initial_state["model"]

        # Configurable callbacks
        self.convert_to_llm = options.convert_to_llm or self._default_convert_to_llm
        self.transform_context = options.transform_context
        self.stream_fn = options.stream_fn or stream_simple
        self.get_api_key = options.get_api_key
        self.on_payload = options.on_payload
        self.on_response = options.on_response
        self.before_tool_call = options.before_tool_call
        self.after_tool_call = options.after_tool_call

        self.session_id = options.session_id
        self.thinking_budgets = options.thinking_budgets
        self.transport = options.transport
        self.max_retry_delay_ms = options.max_retry_delay_ms
        self.tool_execution = options.tool_execution

    def _default_convert_to_llm(self, messages: List[AgentMessage]) -> List[Any]:
        """Default message conversion."""
        # Filter messages by valid roles - use attribute access for Pydantic models
        valid_roles = ("user", "assistant", "toolResult")
        return [m for m in messages if hasattr(m, 'role') and m.role in valid_roles]

    @property
    def state(self) -> AgentState:
        """Get read-only state snapshot."""
        return AgentState(
            system_prompt=self._state.system_prompt,
            model=self._state.model,
            thinking_level=self._state.thinking_level,
            tools=self._state.tools,
            messages=self._state.messages,
            is_streaming=self._state.is_streaming,
            streaming_message=self._state.streaming_message,
            pending_tool_calls=frozenset(self._state.pending_tool_calls),
            error_message=self._state.error_message,
        )

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self._state.system_prompt = prompt

    def set_model(self, model: Model) -> None:
        """Set the model."""
        self._state.model = model

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set the thinking level."""
        self._state.thinking_level = level

    def set_tools(self, tools: List[AgentTool]) -> None:
        """Set the tools list."""
        self._state.tools = tools

    def set_messages(self, messages: List[AgentMessage]) -> None:
        """Set the messages list."""
        self._state.messages = messages