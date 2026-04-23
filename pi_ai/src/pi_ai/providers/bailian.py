"""
Bailian provider implementation.

Provides streaming access to Alibaba's Bailian/Qwen models through the
OpenAI-compatible API endpoint.

Key features:
- Uses httpx for HTTP streaming
- Supports reasoning models (QwQ, Qwen3) with enable_thinking
- Handles reasoning_content field in responses
- Supports tool calling
"""

import json
import os
from typing import Any

import httpx

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


class BailianProvider(ProviderInterface):
    """
    Provider for Alibaba's Bailian/Qwen models.

    Uses the OpenAI-compatible API endpoint at dashscope.aliyuncs.com.
    Supports reasoning models through the enable_thinking parameter.
    """

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    API_KEY_ENV = "DASHSCOPE_API_KEY"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize the Bailian provider.

        Args:
            api_key: Optional API key. If not provided, uses DASHSCOPE_API_KEY env var.
            base_url: Optional base URL override. Defaults to Bailian BASE_URL.
        """
        self.api_key = api_key or os.environ.get(self.API_KEY_ENV)
        if not self.api_key:
            raise ValueError(
                f"API key required. Set {self.API_KEY_ENV} environment variable "
                "or pass api_key parameter."
            )
        self.base_url = base_url or self.BASE_URL

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _convert_message(self, message: UserMessage | AssistantMsg | ToolResultMessage) -> dict[str, Any]:
        """
        Convert a pi-ai message to Bailian/OpenAI format.

        Args:
            message: The message to convert.

        Returns:
            A dictionary in Bailian/OpenAI message format.
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
            for block in message.content:
                if isinstance(block, TextContent):
                    content_parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ThinkingContent):
                    # Bailian uses reasoning_content for thinking
                    content_parts.append({
                        "type": "text",
                        "text": f"<thinking>{block.thinking}</thinking>"
                    })
                elif isinstance(block, ToolCall):
                    content_parts.append({
                        "type": "text",
                        "text": f"[Tool call: {block.name}]"
                    })
            return {"role": "assistant", "content": content_parts}

        elif isinstance(message, ToolResultMessage):
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
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": content_parts
            }

        raise ValueError(f"Unknown message type: {type(message)}")

    def _convert_tool(self, tool: Tool) -> dict[str, Any]:
        """
        Convert a pi-ai Tool to Bailian/OpenAI function format.

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

    def _build_request_body(
        self,
        context: Context,
        model: Model,
        stream: bool = True,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        """
        Build the request body for the API call.

        Args:
            context: The conversation context.
            model: The model configuration.
            stream: Whether to stream the response.
            enable_thinking: Whether to enable thinking for reasoning models.

        Returns:
            The request body as a dictionary.
        """
        messages = []

        # Add system prompt if present
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})

        # Convert messages
        for msg in context.messages:
            messages.append(self._convert_message(msg))

        body: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "stream": stream,
        }

        # Add tools if present
        if context.tools:
            body["tools"] = [self._convert_tool(t) for t in context.tools]

        # Add enable_thinking for reasoning models
        if enable_thinking:
            body["enable_thinking"] = True

        return body

    async def stream(
        self,
        context: Context,
        model: Model,
        enable_thinking: bool = False,
    ) -> AssistantMessageEventStream:
        """
        Stream assistant message events from the Bailian API.

        Args:
            context: The conversation context including messages and tools.
            model: The model configuration to use.
            enable_thinking: Whether to enable thinking mode for reasoning models.

        Returns:
            An AssistantMessageEventStream for iterating over response events.
        """
        stream_result = AssistantMessageEventStream()

        body = self._build_request_body(
            context, model, stream=True, enable_thinking=enable_thinking
        )

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=body,
                    timeout=120.0,
                ) as response:
                    response.raise_for_status()

                    # Emit start event
                    stream_result.push(StartEvent(
                        type="start",
                        api=model.api,
                        provider=model.provider,
                        model=model.id,
                    ))

                    # Track state for streaming
                    text_index = 0
                    thinking_index = 0
                    tool_call_index = 0
                    tool_call_buffers: dict[int, dict[str, str]] = {}
                    usage_data: dict[str, int] = {}
                    response_id: str | None = None
                    finish_reason: str = "stop"

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Capture response ID
                        if "id" in data:
                            response_id = data["id"]

                        # Process choices
                        choices = data.get("choices", [])
                        for choice in choices:
                            delta = choice.get("delta", {})
                            finish = choice.get("finish_reason")

                            if finish:
                                finish_reason = finish

                            # Handle reasoning_content (thinking)
                            if "reasoning_content" in delta:
                                reasoning = delta["reasoning_content"]
                                if reasoning:
                                    if thinking_index == 0:
                                        stream_result.push(ThinkingStartEvent(
                                            type="thinkingStart",
                                            index=thinking_index,
                                        ))
                                    stream_result.push(ThinkingDeltaEvent(
                                        type="thinkingDelta",
                                        index=thinking_index,
                                        delta=reasoning,
                                    ))

                            # Handle regular content
                            if "content" in delta:
                                content = delta["content"]
                                if content:
                                    stream_result.push(TextStartEvent(
                                        type="textStart",
                                        index=text_index,
                                    ))
                                    stream_result.push(TextDeltaEvent(
                                        type="textDelta",
                                        index=text_index,
                                        delta=content,
                                    ))
                                    stream_result.push(TextEndEvent(
                                        type="textEnd",
                                        index=text_index,
                                        text_signature=None,
                                    ))
                                    text_index += 1

                            # Handle tool calls
                            if "tool_calls" in delta:
                                for tc in delta["tool_calls"]:
                                    idx = tc.get("index", 0)
                                    tc_id = tc.get("id", "")
                                    tc_name = tc.get("function", {}).get("name", "")
                                    tc_args = tc.get("function", {}).get("arguments", "")

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

                        # Capture usage from response
                        if "usage" in data:
                            usage_data = data["usage"]

                    # Finalize thinking if any
                    if thinking_index > 0:
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

            except httpx.HTTPStatusError as e:
                stream_result.push(ErrorEvent(
                    type="error",
                    error_message=f"HTTP error {e.response.status_code}: {e.response.text}",
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
        Simple streaming wrapper that automatically enables thinking for reasoning models.

        Args:
            context: The conversation context.
            model: The model configuration.

        Returns:
            An AssistantMessageEventStream for iterating over response events.
        """
        # Automatically enable thinking for reasoning models
        enable_thinking = model.reasoning
        return await self.stream(context, model, enable_thinking=enable_thinking)