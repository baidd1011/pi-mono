"""
OpenAI provider implementation.

Provides streaming access to OpenAI models (GPT-4, GPT-4o, O1, etc.)
using the official OpenAI SDK.

Key features:
- Uses official openai SDK with AsyncOpenAI client
- Supports reasoning models with reasoning_effort parameter
- Handles tool choice options
- Proper error handling
"""

import json
import os
from typing import Any, Literal

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from pi_ai.registry import ProviderInterface
from pi_ai.stream import AssistantMessageEventStream
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    Tool,
    UserMessage,
    AssistantMessage as AssistantMsg,
    ToolResultMessage,
    TextContent,
    ImageContent,
    ThinkingContent,
    ToolCall,
    Usage,
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
)


# Reasoning effort levels for O1-style models
ReasoningEffort = Literal["low", "medium", "high"]


class OpenAIProvider(ProviderInterface):
    """
    Provider for OpenAI models.

    Uses the official OpenAI SDK for API access.
    Supports reasoning models through the reasoning_effort parameter.
    """

    API_KEY_ENV = "OPENAI_API_KEY"
    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: Optional API key. If not provided, uses OPENAI_API_KEY env var.
            base_url: Optional base URL override. Defaults to OpenAI API URL.
        """
        self.api_key = api_key or os.environ.get(self.API_KEY_ENV)
        if not self.api_key:
            raise ValueError(
                f"API key required. Set {self.API_KEY_ENV} environment variable "
                "or pass api_key parameter."
            )
        self.base_url = base_url or self.BASE_URL
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the AsyncOpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _convert_message(
        self, message: UserMessage | AssistantMsg | ToolResultMessage
    ) -> dict[str, Any]:
        """
        Convert a pi-ai message to OpenAI format.

        Args:
            message: The message to convert.

        Returns:
            A dictionary in OpenAI message format.
        """
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                return {"role": "user", "content": message.content}
            else:
                # Handle list of content blocks
                content_parts = []
                for block in message.content:
                    if isinstance(block, TextContent):
                        content_parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageContent):
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{block.mime_type};base64,{block.data}"
                            }
                        })
                return {"role": "user", "content": content_parts}

        elif isinstance(message, AssistantMsg):
            content_parts = []
            tool_calls = []
            for block in message.content:
                if isinstance(block, TextContent):
                    content_parts.append(block.text)
                elif isinstance(block, ThinkingContent):
                    # OpenAI doesn't have native thinking, store as text
                    content_parts.append(f"[Thinking: {block.thinking}]")
                elif isinstance(block, ToolCall):
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.arguments)
                        }
                    })

            result: dict[str, Any] = {"role": "assistant"}
            if content_parts:
                result["content"] = "\n".join(content_parts)
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result

        elif isinstance(message, ToolResultMessage):
            content_parts = []
            for block in message.content:
                if isinstance(block, TextContent):
                    content_parts.append(block.text)
                elif isinstance(block, ImageContent):
                    content_parts.append(f"[Image: {block.mime_type}]")
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": "\n".join(content_parts) if content_parts else ""
            }

        raise ValueError(f"Unknown message type: {type(message)}")

    def _convert_tool(self, tool: Tool) -> dict[str, Any]:
        """
        Convert a pi-ai Tool to OpenAI function format.

        Args:
            tool: The tool to convert.

        Returns:
            A dictionary in OpenAI function format.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_json_schema()
            }
        }

    async def stream(
        self,
        context: Context,
        model: Model,
        reasoning_effort: ReasoningEffort | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AssistantMessageEventStream:
        """
        Stream assistant message events from the OpenAI API.

        Args:
            context: The conversation context including messages and tools.
            model: The model configuration to use.
            reasoning_effort: Optional reasoning effort for O1-style models.
            tool_choice: Optional tool choice setting.

        Returns:
            An AssistantMessageEventStream for iterating over response events.
        """
        stream_result = AssistantMessageEventStream()

        # Build messages
        messages: list[dict[str, Any]] = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})

        for msg in context.messages:
            messages.append(self._convert_message(msg))

        # Build request parameters
        params: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "stream": True,
        }

        # Add tools if present
        if context.tools:
            params["tools"] = [self._convert_tool(t) for t in context.tools]
            if tool_choice:
                params["tool_choice"] = tool_choice

        # Add reasoning_effort for reasoning models
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort

        try:
            # Emit start event
            stream_result.push(StartEvent(
                type="start",
                api=model.api,
                provider=model.provider,
                model=model.id,
            ))

            # Track state for streaming
            text_index = 0
            text_started = False
            thinking_index = 0
            thinking_started = False
            tool_call_index = 0
            tool_call_buffers: dict[int, dict[str, str]] = {}
            usage_data: dict[str, int] = {}
            response_id: str | None = None
            finish_reason: str = "stop"

            # Stream the response
            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                # Capture response ID
                if chunk.id:
                    response_id = chunk.id

                # Capture usage if available
                if chunk.usage:
                    usage_data = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

                # Process choices
                for choice in chunk.choices:
                    delta = choice.delta
                    finish = choice.finish_reason

                    if finish:
                        finish_reason = finish

                    # Handle content
                    if delta.content:
                        if not text_started:
                            stream_result.push(TextStartEvent(
                                type="textStart",
                                index=text_index,
                            ))
                            text_started = True
                        stream_result.push(TextDeltaEvent(
                            type="textDelta",
                            index=text_index,
                            delta=delta.content,
                        ))

                    # Handle reasoning content (for o1-style models)
                    # Note: OpenAI SDK may expose this differently, handle generically
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        if not thinking_started:
                            stream_result.push(ThinkingStartEvent(
                                type="thinkingStart",
                                index=thinking_index,
                            ))
                            thinking_started = True
                        stream_result.push(ThinkingDeltaEvent(
                            type="thinkingDelta",
                            index=thinking_index,
                            delta=delta.reasoning_content,
                        ))

                    # Handle tool calls
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            tc_id = tc.id or ""
                            tc_name = tc.function.name if tc.function else ""
                            tc_args = tc.function.arguments if tc.function else ""

                            if idx not in tool_call_buffers:
                                tool_call_buffers[idx] = {"id": "", "name": "", "arguments": ""}
                                stream_result.push(ToolCallStartEvent(
                                    type="toolCallStart",
                                    index=idx,
                                    id=tc_id,
                                    name=tc_name,
                                ))

                            if tc_id:
                                tool_call_buffers[idx]["id"] = tc_id
                            if tc_name:
                                tool_call_buffers[idx]["name"] = tc_name
                            if tc_args:
                                tool_call_buffers[idx]["arguments"] += tc_args
                                stream_result.push(ToolCallDeltaEvent(
                                    type="toolCallDelta",
                                    index=idx,
                                    id=tool_call_buffers[idx]["id"],
                                    delta=tc_args,
                                ))

            # Finalize text
            if text_started:
                stream_result.push(TextEndEvent(
                    type="textEnd",
                    index=text_index,
                    text_signature=None,
                ))

            # Finalize thinking
            if thinking_started:
                stream_result.push(ThinkingEndEvent(
                    type="thinkingEnd",
                    index=thinking_index,
                    thinking_signature=None,
                ))

            # Finalize tool calls
            for idx, tc_info in sorted(tool_call_buffers.items()):
                try:
                    args = json.loads(tc_info["arguments"])
                except json.JSONDecodeError:
                    args = {}
                stream_result.push(ToolCallEndEvent(
                    type="toolCallEnd",
                    index=idx,
                    id=tc_info["id"],
                    arguments=args,
                ))

            # Emit done event
            usage = Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                cache_read_tokens=0,
                cache_write_tokens=0,
                total_tokens=usage_data.get("total_tokens", 0),
                cost=Usage.Cost(
                    input=model.cost.input,
                    output=model.cost.output,
                    total=model.cost.total,
                )
            )
            stream_result.push(DoneEvent(
                type="done",
                response_id=response_id,
                usage=usage,
                stop_reason="toolUse" if finish_reason == "tool_calls" else finish_reason,
            ))

        except Exception as e:
            stream_result.push(ErrorEvent(
                type="error",
                error_message=str(e),
            ))

        # Build and end the stream
        message = stream_result._build_message()
        stream_result.end(message)
        return stream_result

    async def stream_simple(
        self,
        context: Context,
        model: Model,
    ) -> AssistantMessageEventStream:
        """
        Simple streaming wrapper that automatically sets reasoning_effort for reasoning models.

        Args:
            context: The conversation context.
            model: The model configuration.

        Returns:
            An AssistantMessageEventStream for iterating over response events.
        """
        # Set medium reasoning effort for reasoning models
        reasoning_effort: ReasoningEffort | None = None
        if model.reasoning:
            reasoning_effort = "medium"

        return await self.stream(context, model, reasoning_effort=reasoning_effort)