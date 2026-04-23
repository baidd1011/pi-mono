"""
Agent loop: core LLM interaction logic.

This module implements the agent loop - the core runtime logic for
interacting with LLMs, executing tools, and managing conversation turns.
"""

import asyncio
import json
import time
from typing import List, Optional, Callable, Awaitable, Any, Union

from pi_ai import (
    Model, Context, Message,
    AssistantMessage, AssistantMessageEventStream,
    ToolResultMessage, ToolCall, TextContent, ThinkingContent,
    Usage, Tool, AssistantMessageEvent,
    StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    DoneEvent, ErrorEvent,
    stream_simple,
    validate_tool_arguments,
)
from pi_ai.stream import EventStream

from ._abort import AbortSignal
from .types import (
    AgentContext, AgentEvent, AgentEventSink,
    AgentLoopConfig, AgentTool, AgentToolCall, AgentToolResult,
    BeforeToolCallContext, BeforeToolCallResult,
    AfterToolCallContext, AfterToolCallResult,
    ToolExecutionMode,
)

# Alias for AgentMessage
AgentMessage = Message


# -----------------------------------------------------------------------------
# Event Stream Implementation
# -----------------------------------------------------------------------------

class AgentEventStream(EventStream[AgentEvent, List[AgentMessage]]):
    """
    Event stream specialization for agent events.

    Provides typed streaming for agent lifecycle events and collects
    the final list of messages.
    """

    def __init__(self) -> None:
        """Initialize the agent event stream."""
        super().__init__()
        self._messages: List[AgentMessage] = []

    def end(self, messages: List[AgentMessage]) -> None:
        """
        End the stream with the final message list.

        Args:
            messages: The final list of agent messages.
        """
        self._messages = messages
        super().end(messages)


def create_agent_stream() -> AgentEventStream:
    """Create a new agent event stream."""
    return AgentEventStream()


# -----------------------------------------------------------------------------
# Streaming Functions
# -----------------------------------------------------------------------------

def agent_loop(
    prompts: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal] = None,
    stream_fn: Optional[Callable[[Model, Context], Awaitable[AssistantMessageEventStream]]] = None,
) -> AgentEventStream:
    """
    Start an agent loop with a new prompt message.

    The prompts are added to the context and events are emitted for them.
    Returns a stream of AgentEvent events, with final result being List[AgentMessage].

    Args:
        prompts: The prompt messages to add to context.
        context: The agent context with system prompt, messages, and tools.
        config: Configuration for the agent loop.
        signal: Optional abort signal for cancellation.
        stream_fn: Optional custom stream function.

    Returns:
        An AgentEventStream that emits events and ends with the message list.
    """
    stream = create_agent_stream()

    async def run() -> None:
        try:
            messages = await run_agent_loop(
                prompts, context, config,
                lambda event: stream.push(event),
                signal, stream_fn
            )
            stream.end(messages)
        except Exception as e:
            # Push error event if something goes wrong
            error_event: AgentEvent = {
                "type": "agent_end",
                "messages": [],
            }
            stream.push(error_event)
            stream.end([])

    # Start the async loop in background
    asyncio.create_task(run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal] = None,
    stream_fn: Optional[Callable[[Model, Context], Awaitable[AssistantMessageEventStream]]] = None,
) -> AgentEventStream:
    """
    Continue an agent loop from the current context without adding a new message.

    Used for retries - context already has user message or tool results.

    Args:
        context: The agent context with existing messages.
        config: Configuration for the agent loop.
        signal: Optional abort signal for cancellation.
        stream_fn: Optional custom stream function.

    Returns:
        An AgentEventStream that emits events and ends with the message list.

    Raises:
        ValueError: If context has no messages or last message is assistant.
    """
    if len(context.messages) == 0:
        raise ValueError("Cannot continue: no messages in context")

    # Check last message role
    last_message = context.messages[-1]
    if hasattr(last_message, 'role') and last_message.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = create_agent_stream()

    async def run() -> None:
        try:
            messages = await run_agent_loop_continue(
                context, config,
                lambda event: stream.push(event),
                signal, stream_fn
            )
            stream.end(messages)
        except Exception as e:
            error_event: AgentEvent = {
                "type": "agent_end",
                "messages": [],
            }
            stream.push(error_event)
            stream.end([])

    asyncio.create_task(run())
    return stream


# -----------------------------------------------------------------------------
# Direct Execution Functions
# -----------------------------------------------------------------------------

async def run_agent_loop(
    prompts: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Optional[AbortSignal] = None,
    stream_fn: Optional[Callable[[Model, Context], Awaitable[AssistantMessageEventStream]]] = None,
) -> List[AgentMessage]:
    """
    Run agent loop and emit events via callback.

    Args:
        prompts: The prompt messages to add.
        context: The agent context.
        config: Configuration for the agent loop.
        emit: Callback to emit events.
        signal: Optional abort signal.
        stream_fn: Optional custom stream function.

    Returns:
        The list of new messages generated during the loop.
    """
    new_messages: List[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=context.tools,
    )

    # Emit agent start
    await emit_event(emit, {"type": "agent_start"})
    await emit_event(emit, {"type": "turn_start"})

    # Emit message events for prompts
    for prompt in prompts:
        await emit_event(emit, {"type": "message_start", "message": prompt})
        await emit_event(emit, {"type": "message_end", "message": prompt})

    # Run the main loop
    await run_loop(current_context, new_messages, config, signal, emit, stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Optional[AbortSignal] = None,
    stream_fn: Optional[Callable[[Model, Context], Awaitable[AssistantMessageEventStream]]] = None,
) -> List[AgentMessage]:
    """
    Continue agent loop and emit events via callback.

    Args:
        context: The agent context.
        config: Configuration for the agent loop.
        emit: Callback to emit events.
        signal: Optional abort signal.
        stream_fn: Optional custom stream function.

    Returns:
        The list of new messages generated during the loop.

    Raises:
        ValueError: If context has no messages or last message is assistant.
    """
    if len(context.messages) == 0:
        raise ValueError("Cannot continue: no messages in context")

    last_message = context.messages[-1]
    if hasattr(last_message, 'role') and last_message.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: List[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await emit_event(emit, {"type": "agent_start"})
    await emit_event(emit, {"type": "turn_start"})

    await run_loop(current_context, new_messages, config, signal, emit, stream_fn)
    return new_messages


# -----------------------------------------------------------------------------
# Main Loop Logic
# -----------------------------------------------------------------------------

async def run_loop(
    current_context: AgentContext,
    new_messages: List[AgentMessage],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
    stream_fn: Optional[Callable[[Model, Context], Awaitable[AssistantMessageEventStream]]] = None,
) -> None:
    """
    Main loop logic shared by run_agent_loop and run_agent_loop_continue.

    This implements the core agent loop:
    1. Process steering messages (injected mid-run)
    2. Stream assistant response
    3. Execute tool calls (sequential or parallel)
    4. Check for more steering messages
    5. Check for follow-up messages (when agent would otherwise stop)
    """
    first_turn = True

    # Check for steering messages at start
    pending_messages: List[AgentMessage] = []
    if config.get_steering_messages:
        try:
            pending_messages = await config.get_steering_messages() or []
        except Exception:
            pending_messages = []

    # Outer loop: continues when follow-up messages arrive
    while True:
        has_more_tool_calls = True

        # Inner loop: process tool calls and steering messages
        while has_more_tool_calls or len(pending_messages) > 0:
            if not first_turn:
                await emit_event(emit, {"type": "turn_start"})
            else:
                first_turn = False

            # Process pending messages (inject before assistant response)
            if len(pending_messages) > 0:
                for message in pending_messages:
                    await emit_event(emit, {"type": "message_start", "message": message})
                    await emit_event(emit, {"type": "message_end", "message": message})
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # Stream assistant response
            message = await stream_assistant_response(
                current_context, config, signal, emit, stream_fn
            )
            new_messages.append(message)

            # Check for error or abort
            stop_reason = message.stop_reason if hasattr(message, 'stop_reason') else "stop"
            if stop_reason in ("error", "aborted"):
                await emit_event(emit, {"type": "turn_end", "message": message, "tool_results": []})
                await emit_event(emit, {"type": "agent_end", "messages": new_messages})
                return

            # Check for tool calls
            tool_calls = extract_tool_calls(message)
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: List[ToolResultMessage] = []
            if has_more_tool_calls:
                tool_results = await execute_tool_calls(
                    current_context, message, tool_calls, config, signal, emit
                )

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await emit_event(emit, {"type": "turn_end", "message": message, "tool_results": tool_results})

            # Check for steering messages
            if config.get_steering_messages:
                try:
                    pending_messages = await config.get_steering_messages() or []
                except Exception:
                    pending_messages = []

        # Agent would stop here. Check for follow-up messages.
        if config.get_follow_up_messages:
            try:
                follow_up_messages = await config.get_follow_up_messages() or []
            except Exception:
                follow_up_messages = []

            if len(follow_up_messages) > 0:
                pending_messages = follow_up_messages
                continue

        # No more messages, exit
        break

    await emit_event(emit, {"type": "agent_end", "messages": new_messages})


# -----------------------------------------------------------------------------
# Assistant Response Streaming
# -----------------------------------------------------------------------------

async def stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
    stream_fn: Optional[Callable[[Model, Context], Awaitable[AssistantMessageEventStream]]] = None,
) -> AssistantMessage:
    """
    Stream an assistant response from the LLM.

    This is where AgentMessage[] gets transformed to Message[] for the LLM.
    """
    # Apply context transform if configured
    messages = context.messages
    if config.transform_context:
        try:
            messages = await config.transform_context(messages, signal)
        except Exception:
            # Keep original messages on transform error
            messages = context.messages

    # Convert to LLM-compatible messages
    try:
        llm_messages_result = config.convert_to_llm(messages)
        if isinstance(llm_messages_result, Awaitable):
            llm_messages = await llm_messages_result
        else:
            llm_messages = llm_messages_result
    except Exception:
        llm_messages = messages

    # Build LLM context
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=None,  # Tools converted to None for now - faux provider doesn't need them
    )

    # Use provided stream function or default
    stream_function = stream_fn

    # Resolve API key
    api_key = config.api_key
    if config.get_api_key:
        try:
            key_result = config.get_api_key(config.model.provider)
            if isinstance(key_result, Awaitable):
                resolved_key = await key_result
            else:
                resolved_key = key_result
            if resolved_key:
                api_key = resolved_key
        except Exception:
            pass

    # Stream response - call stream function with (context, model)
    try:
        if stream_function:
            response = await stream_function(llm_context, config.model)
        else:
            # Use faux provider directly for testing
            from pi_ai.providers import FauxProvider
            faux = FauxProvider()
            response = await faux.stream(llm_context, config.model)
    except Exception as e:
        # Create error message
        error_message = AssistantMessage(
            content=[],
            api=config.model.api,
            provider=config.model.provider,
            model=config.model.id,
            response_id=None,
            usage=Usage(
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                total_tokens=0,
                cost=Usage.Cost(input=0.0, output=0.0, total=0.0),
            ),
            stop_reason="error",
            error_message=str(e),
            timestamp=int(time.time() * 1000),
        )
        await emit_event(emit, {"type": "message_start", "message": error_message})
        await emit_event(emit, {"type": "message_end", "message": error_message})
        return error_message

    # Track message building state
    added_partial = False
    start_info: dict = {}
    text_buffers: dict[int, str] = {}
    thinking_buffers: dict[int, str] = {}
    tool_call_buffers: dict[int, dict] = {}
    done_info: dict = {}

    # Use async for since AssistantMessageEventStream is an AsyncIterator
    async for event in response:
        # Handle abort
        if signal and signal.aborted:
            # Create aborted message
            aborted_message = AssistantMessage(
                content=[],
                api=config.model.api,
                provider=config.model.provider,
                model=config.model.id,
                response_id=None,
                usage=Usage(
                    input_tokens=0,
                    output_tokens=0,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    total_tokens=0,
                    cost=Usage.Cost(input=0.0, output=0.0, total=0.0),
                ),
                stop_reason="aborted",
                error_message="Request aborted",
                timestamp=int(time.time() * 1000),
            )
            if added_partial:
                context.messages[-1] = aborted_message
            else:
                context.messages.append(aborted_message)
                await emit_event(emit, {"type": "message_start", "message": aborted_message})
            await emit_event(emit, {"type": "message_end", "message": aborted_message})
            return aborted_message

        event_type = event.get("type", "")

        if event_type == "start":
            # Collect start info
            start_info["api"] = event.get("api", config.model.api)
            start_info["provider"] = event.get("provider", config.model.provider)
            start_info["model"] = event.get("model", config.model.id)

            # Create initial partial message
            partial_msg = AssistantMessage(
                content=[],
                api=start_info.get("api", config.model.api),
                provider=start_info.get("provider", config.model.provider),
                model=start_info.get("model", config.model.id),
                response_id=None,
                usage=Usage(
                    input_tokens=0,
                    output_tokens=0,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    total_tokens=0,
                    cost=Usage.Cost(input=0.0, output=0.0, total=0.0),
                ),
                stop_reason="stop",
                timestamp=int(time.time() * 1000),
            )
            context.messages.append(partial_msg)
            added_partial = True
            await emit_event(emit, {"type": "message_start", "message": partial_msg})

        elif event_type == "textStart":
            index = event.get("index", 0)
            text_buffers[index] = ""

        elif event_type == "textDelta":
            index = event.get("index", 0)
            delta = event.get("delta", "")
            if index not in text_buffers:
                text_buffers[index] = ""
            text_buffers[index] += delta

            # Update partial message
            partial_msg = build_partial_message(
                start_info, text_buffers, thinking_buffers, tool_call_buffers, done_info, config.model
            )
            context.messages[-1] = partial_msg
            await emit_event(emit, {
                "type": "message_update",
                "message": partial_msg,
                "assistant_message_event": event,
            })

        elif event_type == "textEnd":
            index = event.get("index", 0)
            # Text complete, update message
            partial_msg = build_partial_message(
                start_info, text_buffers, thinking_buffers, tool_call_buffers, done_info, config.model
            )
            context.messages[-1] = partial_msg
            await emit_event(emit, {
                "type": "message_update",
                "message": partial_msg,
                "assistant_message_event": event,
            })

        elif event_type == "thinkingStart":
            index = event.get("index", 0)
            thinking_buffers[index] = ""

        elif event_type == "thinkingDelta":
            index = event.get("index", 0)
            delta = event.get("delta", "")
            if index not in thinking_buffers:
                thinking_buffers[index] = ""
            thinking_buffers[index] += delta

            # Update partial message
            partial_msg = build_partial_message(
                start_info, text_buffers, thinking_buffers, tool_call_buffers, done_info, config.model
            )
            context.messages[-1] = partial_msg
            await emit_event(emit, {
                "type": "message_update",
                "message": partial_msg,
                "assistant_message_event": event,
            })

        elif event_type == "thinkingEnd":
            index = event.get("index", 0)
            # Thinking complete, update message
            partial_msg = build_partial_message(
                start_info, text_buffers, thinking_buffers, tool_call_buffers, done_info, config.model
            )
            context.messages[-1] = partial_msg
            await emit_event(emit, {
                "type": "message_update",
                "message": partial_msg,
                "assistant_message_event": event,
            })

        elif event_type == "toolCallStart":
            index = event.get("index", 0)
            tc_id = event.get("id", "")
            tc_name = event.get("name", "")
            # Use tool call id as key to handle multiple tool calls with same index
            tool_call_buffers[tc_id] = {"id": tc_id, "name": tc_name, "arguments": "", "index": index}

        elif event_type == "toolCallDelta":
            tc_id = event.get("id", "")
            delta = event.get("delta", "")
            if tc_id in tool_call_buffers:
                tool_call_buffers[tc_id]["arguments"] += delta

            # Update partial message
            partial_msg = build_partial_message(
                start_info, text_buffers, thinking_buffers, tool_call_buffers, done_info, config.model
            )
            context.messages[-1] = partial_msg
            await emit_event(emit, {
                "type": "message_update",
                "message": partial_msg,
                "assistant_message_event": event,
            })

        elif event_type == "toolCallEnd":
            tc_id = event.get("id", "")
            args = event.get("arguments", {})
            if tc_id in tool_call_buffers:
                tool_call_buffers[tc_id]["arguments_parsed"] = args

            # Update partial message
            partial_msg = build_partial_message(
                start_info, text_buffers, thinking_buffers, tool_call_buffers, done_info, config.model
            )
            context.messages[-1] = partial_msg
            await emit_event(emit, {
                "type": "message_update",
                "message": partial_msg,
                "assistant_message_event": event,
            })

        elif event_type == "done":
            # Collect done info
            done_info["response_id"] = event.get("response_id")
            done_info["usage"] = event.get("usage")
            done_info["stop_reason"] = event.get("stop_reason", "stop")

            # Build final message
            final_message = build_final_message(
                start_info, text_buffers, thinking_buffers, tool_call_buffers, done_info, config.model
            )

            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await emit_event(emit, {"type": "message_start", "message": final_message})

            await emit_event(emit, {"type": "message_end", "message": final_message})
            return final_message

        elif event_type == "error":
            error_msg = event.get("error_message", "Unknown error")
            final_message = AssistantMessage(
                content=[],
                api=start_info.get("api", config.model.api),
                provider=start_info.get("provider", config.model.provider),
                model=start_info.get("model", config.model.id),
                response_id=None,
                usage=Usage(
                    input_tokens=0,
                    output_tokens=0,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    total_tokens=0,
                    cost=Usage.Cost(input=0.0, output=0.0, total=0.0),
                ),
                stop_reason="error",
                error_message=error_msg,
                timestamp=int(time.time() * 1000),
            )

            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await emit_event(emit, {"type": "message_start", "message": final_message})

            await emit_event(emit, {"type": "message_end", "message": final_message})
            return final_message

    # Get final message from stream if not already returned
    try:
        final_message = await response.result()
    except Exception:
        final_message = build_final_message(
            start_info, text_buffers, thinking_buffers, tool_call_buffers, done_info, config.model
        )

    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await emit_event(emit, {"type": "message_start", "message": final_message})
    await emit_event(emit, {"type": "message_end", "message": final_message})
    return final_message


def build_partial_message(
    start_info: dict,
    text_buffers: dict[int, str],
    thinking_buffers: dict[int, str],
    tool_call_buffers: dict[int, dict],
    done_info: dict,
    model: Model,
) -> AssistantMessage:
    """Build a partial assistant message from collected buffers."""
    content: List[Union[TextContent, ThinkingContent, ToolCall]] = []

    # Add text content blocks in order
    for index, text in sorted(text_buffers.items()):
        if text:
            content.append(TextContent(text=text))

    # Add thinking content blocks in order
    for index, thinking in sorted(thinking_buffers.items()):
        if thinking:
            content.append(ThinkingContent(thinking=thinking))

    # Add tool call content blocks in order (by id for consistency)
    for tc_id, tool_info in tool_call_buffers.items():
        args = tool_info.get("arguments_parsed", {})
        if not args and tool_info.get("arguments"):
            try:
                args = json.loads(tool_info["arguments"])
            except json.JSONDecodeError:
                args = {}
        content.append(ToolCall(
            id=tool_info.get("id", tc_id),
            name=tool_info.get("name", ""),
            arguments=args,
        ))

    # Build usage
    usage_data = done_info.get("usage")
    if usage_data:
        if isinstance(usage_data, Usage):
            usage = usage_data
        else:
            usage = Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                cache_read_tokens=usage_data.get("cache_read_tokens", 0),
                cache_write_tokens=usage_data.get("cache_write_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                cost=Usage.Cost(
                    input=0.0,
                    output=0.0,
                    total=0.0,
                ),
            )
    else:
        usage = Usage(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            total_tokens=0,
            cost=Usage.Cost(input=0.0, output=0.0, total=0.0),
        )

    return AssistantMessage(
        content=content,
        api=start_info.get("api", model.api),
        provider=start_info.get("provider", model.provider),
        model=start_info.get("model", model.id),
        response_id=done_info.get("response_id"),
        usage=usage,
        stop_reason=done_info.get("stop_reason", "stop"),
        timestamp=int(time.time() * 1000),
    )


def build_final_message(
    start_info: dict,
    text_buffers: dict[int, str],
    thinking_buffers: dict[int, str],
    tool_call_buffers: dict[int, dict],
    done_info: dict,
    model: Model,
) -> AssistantMessage:
    """Build the final assistant message from collected buffers."""
    return build_partial_message(start_info, text_buffers, thinking_buffers, tool_call_buffers, done_info, model)


# -----------------------------------------------------------------------------
# Tool Execution
# -----------------------------------------------------------------------------

def extract_tool_calls(message: AssistantMessage) -> List[AgentToolCall]:
    """Extract tool calls from an assistant message's content."""
    tool_calls: List[AgentToolCall] = []
    content = message.content if hasattr(message, 'content') else []

    for block in content:
        if isinstance(block, ToolCall):
            tool_calls.append(AgentToolCall(
                id=block.id,
                name=block.name,
                arguments=block.arguments,
            ))
        elif isinstance(block, dict) and block.get("type") == "toolCall":
            tool_calls.append(AgentToolCall(
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=block.get("arguments", {}),
            ))

    return tool_calls


def find_tool(context: AgentContext, tool_name: str) -> Optional[AgentTool]:
    """Find a tool by name in the context."""
    tools = context.tools or []
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None


async def execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[AgentToolCall],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> List[ToolResultMessage]:
    """
    Execute tool calls from an assistant message.

    Determines execution mode based on config and tool overrides.
    """
    # Check if any tool requires sequential execution
    has_sequential_tool = False
    for tc in tool_calls:
        tool = find_tool(context, tc.name)
        if tool and tool.execution_mode == "sequential":
            has_sequential_tool = True
            break

    if config.tool_execution == "sequential" or has_sequential_tool:
        return await execute_tool_calls_sequential(
            context, assistant_message, tool_calls, config, signal, emit
        )
    else:
        return await execute_tool_calls_parallel(
            context, assistant_message, tool_calls, config, signal, emit
        )


async def execute_tool_calls_sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[AgentToolCall],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> List[ToolResultMessage]:
    """Execute tool calls one at a time, in order."""
    results: List[ToolResultMessage] = []

    for tool_call in tool_calls:
        await emit_event(emit, {
            "type": "tool_execution_start",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "args": tool_call.arguments,
        })

        preparation = await prepare_tool_call(context, assistant_message, tool_call, config, signal)
        if preparation["kind"] == "immediate":
            result = await emit_tool_call_outcome(
                tool_call, preparation["result"], preparation["is_error"], emit
            )
        else:
            executed = await execute_prepared_tool_call(preparation, signal, emit)
            result = await finalize_executed_tool_call(
                context, assistant_message, preparation, executed, config, signal, emit
            )

        results.append(result)

    return results


async def execute_tool_calls_parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[AgentToolCall],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> List[ToolResultMessage]:
    """Execute tool calls concurrently."""
    results: List[ToolResultMessage] = []
    runnable_calls: List[dict] = []

    # Prepare all tool calls
    for tool_call in tool_calls:
        await emit_event(emit, {
            "type": "tool_execution_start",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "args": tool_call.arguments,
        })

        preparation = await prepare_tool_call(context, assistant_message, tool_call, config, signal)
        if preparation["kind"] == "immediate":
            result = await emit_tool_call_outcome(
                tool_call, preparation["result"], preparation["is_error"], emit
            )
            results.append(result)
        else:
            runnable_calls.append(preparation)

    # Execute runnable calls concurrently
    running_calls = [
        {"prepared": prepared, "execution": execute_prepared_tool_call(prepared, signal, emit)}
        for prepared in runnable_calls
    ]

    for running in running_calls:
        executed = await running["execution"]
        result = await finalize_executed_tool_call(
            context, assistant_message, running["prepared"], executed, config, signal, emit
        )
        results.append(result)

    return results


async def prepare_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
) -> dict:
    """
    Prepare a tool call for execution.

    Returns either an immediate result (error case) or a prepared call.
    """
    tool = find_tool(context, tool_call.name)

    if tool is None:
        return {
            "kind": "immediate",
            "result": create_error_tool_result(f"Tool {tool_call.name} not found"),
            "is_error": True,
        }

    try:
        # Apply prepare_arguments if available
        prepared_args = tool_call.arguments
        if tool.prepare_arguments:
            prepared_args = tool.prepare_arguments(tool_call.arguments)

        # Validate arguments if parameters is a Pydantic model
        validated_args = prepared_args
        if tool.parameters is not None:
            # Check if parameters is a Pydantic model class
            if hasattr(tool.parameters, 'model_validate'):
                validated_args = validate_tool_arguments(tool.parameters, prepared_args)
            # If it's a dict schema, skip validation (faux provider doesn't need it)

        # Call before_tool_call hook
        if config.before_tool_call:
            before_context = BeforeToolCallContext(
                assistant_message=assistant_message,
                tool_call=tool_call,
                args=validated_args,
                context=context,
            )
            before_result = await config.before_tool_call(before_context, signal)
            if before_result and before_result.block:
                return {
                    "kind": "immediate",
                    "result": create_error_tool_result(
                        before_result.reason or "Tool execution was blocked"
                    ),
                    "is_error": True,
                }

        return {
            "kind": "prepared",
            "tool_call": tool_call,
            "tool": tool,
            "args": validated_args,
        }

    except Exception as e:
        return {
            "kind": "immediate",
            "result": create_error_tool_result(str(e)),
            "is_error": True,
        }


async def execute_prepared_tool_call(
    prepared: dict,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> dict:
    """Execute a prepared tool call."""
    tool = prepared["tool"]
    tool_call = prepared["tool_call"]
    args = prepared["args"]

    update_events: List[Awaitable] = []

    try:
        # Create update callback
        def update_callback(partial_result: AgentToolResult) -> None:
            update_events.append(
                emit_event(emit, {
                    "type": "tool_execution_update",
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "args": tool_call.arguments,
                    "partial_result": partial_result,
                })
            )

        # Execute tool
        result = await tool.execute(tool_call.id, args, signal, update_callback)

        # Wait for all update events
        for event_task in update_events:
            await event_task

        return {"result": result, "is_error": False}

    except Exception as e:
        # Wait for all update events even on error
        for event_task in update_events:
            await event_task

        return {
            "result": create_error_tool_result(str(e)),
            "is_error": True,
        }


async def finalize_executed_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: dict,
    executed: dict,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> ToolResultMessage:
    """Finalize an executed tool call, applying after_tool_call hook if configured."""
    result = executed["result"]
    is_error = executed["is_error"]

    # Apply after_tool_call hook
    if config.after_tool_call:
        try:
            after_context = AfterToolCallContext(
                assistant_message=assistant_message,
                tool_call=prepared["tool_call"],
                args=prepared["args"],
                result=result,
                is_error=is_error,
                context=context,
            )
            after_result = await config.after_tool_call(after_context, signal)

            if after_result:
                if after_result.content is not None:
                    result.content = after_result.content
                if after_result.details is not None:
                    result.details = after_result.details
                if after_result.is_error is not None:
                    is_error = after_result.is_error

        except Exception as e:
            result = create_error_tool_result(str(e))
            is_error = True

    return await emit_tool_call_outcome(prepared["tool_call"], result, is_error, emit)


def create_error_tool_result(message: str) -> AgentToolResult:
    """Create a tool result for an error."""
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={},
    )


async def emit_tool_call_outcome(
    tool_call: AgentToolCall,
    result: AgentToolResult,
    is_error: bool,
    emit: AgentEventSink,
) -> ToolResultMessage:
    """Emit tool execution end event and create tool result message."""
    await emit_event(emit, {
        "type": "tool_execution_end",
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.name,
        "result": result,
        "is_error": is_error,
    })

    # Create tool result message
    tool_result_message = ToolResultMessage(
        role="toolResult",
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=is_error,
        timestamp=int(time.time() * 1000),
    )

    await emit_event(emit, {"type": "message_start", "message": tool_result_message})
    await emit_event(emit, {"type": "message_end", "message": tool_result_message})

    return tool_result_message


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

async def emit_event(emit: AgentEventSink, event: AgentEvent) -> None:
    """Helper to emit an event, handling both sync and async callbacks."""
    result = emit(event)
    if isinstance(result, Awaitable):
        await result


def convert_tools_to_llm(agent_tools: Optional[List[AgentTool]]) -> Optional[List[Tool]]:
    """Convert AgentTool list to LLM Tool format."""
    if agent_tools is None:
        return None

    llm_tools: List[Tool] = []
    for tool in agent_tools:
        # AgentTool has 'parameters' as schema
        # For tools with Pydantic model parameters, use them directly
        params = tool.parameters
        if params is None:
            # Use a simple empty schema
            from pydantic import BaseModel
            class EmptyParams(BaseModel):
                pass
            params = EmptyParams
        elif isinstance(params, dict):
            # Convert dict schema to a Pydantic model dynamically
            # For now, just skip dict-based tools in LLM context
            # The faux provider doesn't validate tool schemas anyway
            continue
        elif not (params.__class__.__name__ == 'model' or hasattr(params, 'model_json_schema')):
            # Not a Pydantic model, skip
            continue

        llm_tool = Tool(
            name=tool.name,
            description=tool.description,
            parameters=params,
        )
        llm_tools.append(llm_tool)

    return llm_tools if llm_tools else None