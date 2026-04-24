"""Agent class - stateful wrapper around agent loop."""

import asyncio
from typing import Callable, Optional, List, Union, Awaitable, Any, Literal
from dataclasses import dataclass, field

from pi_ai import (
    Model, Context, Message, Usage, UserMessage,
    AssistantMessageEventStream, ImageContent, TextContent,
    stream_simple,
)
from ._abort import AbortController, AbortSignal
from .types import (
    AgentState, AgentContext, AgentTool, AgentToolResult,
    AgentEvent, ToolExecutionMode, ThinkingLevel, AgentLoopConfig,
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

        # Event subscription
        self._listeners: List[Callable[[AgentEvent, AbortSignal], Awaitable[None] | None]] = []

        # Message queues
        self._steering_queue: List[AgentMessage] = []
        self._steering_mode: Literal["all", "one-at-a-time"] = options.steering_mode
        self._follow_up_queue: List[AgentMessage] = []
        self._follow_up_mode: Literal["all", "one-at-a-time"] = options.follow_up_mode

        # Active run tracking
        self._active_run: Optional[dict] = None

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

    # -------------------------------------------------------------------------
    # Event Subscription
    # -------------------------------------------------------------------------

    def subscribe(self, listener: Callable[[AgentEvent, AbortSignal], Awaitable[None] | None]) -> Callable[[], None]:
        """Subscribe to agent events. Returns unsubscribe function."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    async def _emit_event(self, event: AgentEvent) -> None:
        """Emit event to all listeners."""
        signal = self._active_run.get("abort_controller").signal if self._active_run else None
        for listener in self._listeners:
            result = listener(event, signal)
            if isinstance(result, Awaitable):
                await result

    # -------------------------------------------------------------------------
    # Message Queues
    # -------------------------------------------------------------------------

    def steer(self, message: AgentMessage) -> None:
        """Queue message to inject after current assistant turn."""
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue message for after agent would otherwise stop."""
        self._follow_up_queue.append(message)

    def clear_steering_queue(self) -> None:
        """Clear the steering queue."""
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        """Clear the follow-up queue."""
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        """Clear both steering and follow-up queues."""
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        """Check if there are any queued messages."""
        return len(self._steering_queue) > 0 or len(self._follow_up_queue) > 0

    @property
    def steering_mode(self) -> Literal["all", "one-at-a-time"]:
        """Get steering mode."""
        return self._steering_mode

    @steering_mode.setter
    def steering_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        """Set steering mode."""
        self._steering_mode = mode

    @property
    def follow_up_mode(self) -> Literal["all", "one-at-a-time"]:
        """Get follow-up mode."""
        return self._follow_up_mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        """Set follow-up mode."""
        self._follow_up_mode = mode

    def _drain_steering(self) -> List[AgentMessage]:
        """Drain steering queue based on mode."""
        if self._steering_mode == "all":
            result = self._steering_queue.copy()
            self._steering_queue.clear()
            return result

        if self._steering_queue:
            return [self._steering_queue.pop(0)]
        return []

    def _drain_follow_up(self) -> List[AgentMessage]:
        """Drain follow-up queue based on mode."""
        if self._follow_up_mode == "all":
            result = self._follow_up_queue.copy()
            self._follow_up_queue.clear()
            return result

        if self._follow_up_queue:
            return [self._follow_up_queue.pop(0)]
        return []

    # -------------------------------------------------------------------------
    # Abort Handling
    # -------------------------------------------------------------------------

    @property
    def signal(self) -> Optional[AbortSignal]:
        """Active abort signal for current run."""
        return self._active_run.get("abort_controller").signal if self._active_run else None

    def abort(self) -> None:
        """Abort the current run."""
        if self._active_run:
            self._active_run["abort_controller"].abort()

    async def wait_for_idle(self) -> None:
        """Wait for current run to finish."""
        if self._active_run:
            await self._active_run["promise"]

    def reset(self) -> None:
        """Clear state and queues."""
        self._state._messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_all_queues()

    # -------------------------------------------------------------------------
    # Prompt and Continue
    # -------------------------------------------------------------------------

    async def prompt(self, input: Union[str, AgentMessage, List[AgentMessage]], images: Optional[List[ImageContent]] = None) -> None:
        """
        Start a new prompt.

        Args:
            input: The input - can be a string, a single message, or a list of messages.
            images: Optional images to attach to the prompt (only when input is a string).

        Raises:
            RuntimeError: If the agent already has an active run.
        """
        if self._active_run:
            raise RuntimeError("Agent already has an active run")

        # Convert input to messages
        messages: List[AgentMessage] = []
        if isinstance(input, str):
            # Convert string to user message
            content: List[Union[TextContent, ImageContent]] = [TextContent(text=input)]
            if images:
                content.extend(images)
            messages = [UserMessage(role="user", content=content, timestamp=int(asyncio.get_event_loop().time() * 1000))]
        elif isinstance(input, list):
            messages = input
        else:
            # Single message
            messages = [input]

        # Add messages to state
        current_messages = list(self._state.messages) + messages
        self._state.messages = current_messages

        # Create context
        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=current_messages,
            tools=self._state.tools,
        )

        # Create abort controller
        abort_controller = AbortController()

        # Create run promise
        run_complete: asyncio.Future = asyncio.get_event_loop().create_future()

        self._active_run = {
            "abort_controller": abort_controller,
            "promise": run_complete,
        }

        try:
            # Import agent loop functions
            from .agent_loop import run_agent_loop

            # Build config
            config = AgentLoopConfig(
                model=self._state.model,
                convert_to_llm=self.convert_to_llm,
                transform_context=self.transform_context,
                get_api_key=self.get_api_key,
                get_steering_messages=self._get_steering_messages,
                get_follow_up_messages=self._get_follow_up_messages,
                tool_execution=self.tool_execution,
                before_tool_call=self.before_tool_call,
                after_tool_call=self.after_tool_call,
            )

            # Create emit callback
            async def emit(event: AgentEvent) -> None:
                await self._emit_event(event)

            # Run agent loop
            result_messages = await run_agent_loop(
                messages,
                context,
                config,
                emit,
                abort_controller.signal,
                self.stream_fn,
            )

            # Update state with result messages
            all_messages = list(self._state.messages) + result_messages
            self._state.messages = all_messages

            run_complete.set_result(None)

        except Exception as e:
            run_complete.set_exception(e)
            raise
        finally:
            self._active_run = None

    async def continue_(self) -> None:
        """
        Continue from the current transcript.

        The last message must be a user message or tool result message.
        Cannot continue from an assistant message.

        Raises:
            RuntimeError: If the agent already has an active run.
            ValueError: If there are no messages or last message is assistant.
        """
        if self._active_run:
            raise RuntimeError("Agent already has an active run")

        # Validate last message
        current_messages = self._state.messages
        if not current_messages:
            raise ValueError("Cannot continue: no messages")

        last_message = current_messages[-1]
        last_role = getattr(last_message, 'role', None)
        if last_role == "assistant":
            raise ValueError("Cannot continue from assistant message")

        # Create context with existing messages
        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=current_messages,
            tools=self._state.tools,
        )

        # Create abort controller
        abort_controller = AbortController()

        # Create run promise
        run_complete: asyncio.Future = asyncio.get_event_loop().create_future()

        self._active_run = {
            "abort_controller": abort_controller,
            "promise": run_complete,
        }

        try:
            # Import agent loop functions
            from .agent_loop import run_agent_loop_continue

            # Build config
            config = AgentLoopConfig(
                model=self._state.model,
                convert_to_llm=self.convert_to_llm,
                transform_context=self.transform_context,
                get_api_key=self.get_api_key,
                get_steering_messages=self._get_steering_messages,
                get_follow_up_messages=self._get_follow_up_messages,
                tool_execution=self.tool_execution,
                before_tool_call=self.before_tool_call,
                after_tool_call=self.after_tool_call,
            )

            # Create emit callback
            async def emit(event: AgentEvent) -> None:
                await self._emit_event(event)

            # Run agent loop continue
            result_messages = await run_agent_loop_continue(
                context,
                config,
                emit,
                abort_controller.signal,
                self.stream_fn,
            )

            # Update state with result messages
            all_messages = list(self._state.messages) + result_messages
            self._state.messages = all_messages

            run_complete.set_result(None)

        except Exception as e:
            run_complete.set_exception(e)
            raise
        finally:
            self._active_run = None

    async def _get_steering_messages(self) -> List[AgentMessage]:
        """Get messages from the steering queue."""
        return self._drain_steering()

    async def _get_follow_up_messages(self) -> List[AgentMessage]:
        """Get messages from the follow-up queue."""
        return self._drain_follow_up()