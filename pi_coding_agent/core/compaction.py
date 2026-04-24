"""
Context compaction for long sessions.

This module provides functionality to compress old messages into summaries
when the context exceeds a threshold. This helps manage long conversations
that would otherwise exceed the model's context window.
"""

from typing import List, Optional, Tuple, Callable, Union, Any
from datetime import datetime

from pi_ai import UserMessage, AssistantMessage, ToolResultMessage, Message
from pi_ai.types import TextContent, ThinkingContent, ToolCall


def _get_block_type(block: Any) -> str:
    """Get the type of a content block, handling both dicts and Pydantic models."""
    if isinstance(block, dict):
        return block.get("type", "")
    elif hasattr(block, "type"):
        return block.type
    return ""


def _get_block_text(block: Any) -> str:
    """Get text from a content block, handling both dicts and Pydantic models."""
    if isinstance(block, dict):
        return block.get("text", "")
    elif isinstance(block, TextContent):
        return block.text
    return ""


def _get_block_thinking(block: Any) -> str:
    """Get thinking text from a content block."""
    if isinstance(block, dict):
        return block.get("thinking", "")
    elif isinstance(block, ThinkingContent):
        return block.thinking
    return ""


def _get_tool_call_info(block: Any) -> tuple[str, dict]:
    """Get tool call name and arguments from a block."""
    if isinstance(block, dict):
        name = block.get("name", "unknown")
        args = block.get("arguments", {})
        return name, args if isinstance(args, dict) else {}
    elif isinstance(block, ToolCall):
        return block.name, block.arguments
    return "unknown", {}


def estimate_context_tokens(messages: List[Message]) -> int:
    """
    Estimate token count for messages.

    Uses a rough heuristic of 4 characters per token for text content,
    plus overhead for tool calls.

    Args:
        messages: List of messages to estimate tokens for.

    Returns:
        Estimated total token count.
    """
    total = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            total += len(content) // 4  # Rough estimate: 4 chars per token
        elif isinstance(content, list):
            for block in content:
                block_type = _get_block_type(block)
                if block_type == "text":
                    total += len(_get_block_text(block)) // 4
                elif block_type == "thinking":
                    total += len(_get_block_thinking(block)) // 4
                elif block_type == "toolCall":
                    total += 50  # Tool call overhead
                    name, args = _get_tool_call_info(block)
                    total += len(str(args)) // 4

        # Count role and other metadata overhead
        total += 10  # Approximate overhead per message

    return total


def should_compact(
    messages: List[Message],
    context_window: int,
    threshold: float = 0.8,
) -> bool:
    """
    Check if context needs compaction.

    Args:
        messages: List of messages to check.
        context_window: Maximum context window size in tokens.
        threshold: Fraction of context window at which to trigger compaction.

    Returns:
        True if compaction is needed, False otherwise.
    """
    tokens = estimate_context_tokens(messages)
    return tokens > context_window * threshold


def select_compaction_range(
    messages: List[Message],
    keep_recent: int = 10,
    keep_system: bool = True,
) -> Tuple[List[Message], List[Message]]:
    """
    Select messages to compact and keep.

    Args:
        messages: List of messages to partition.
        keep_recent: Number of recent messages to keep.
        keep_system: Whether to preserve the first system message.

    Returns:
        Tuple of (to_compact, to_keep).
    """
    if len(messages) <= keep_recent:
        return [], messages

    # Keep recent messages
    to_keep = list(messages[-keep_recent:])
    to_compact = list(messages[:-keep_recent])

    # Optionally keep first system message
    if keep_system and to_compact and to_compact[0].role == "user":
        # Check if first message is a system-like message (user with system instructions)
        content = to_compact[0].content
        if isinstance(content, str) and len(content) > 100:
            # Assume this might be a system prompt
            pass  # For now, treat normally

    return to_compact, to_keep


async def compact_messages(
    messages: List[Message],
    keep_recent: int = 10,
    context_window: int = 128000,
    summary_fn: Optional[Callable[[List[Message]], str]] = None,
) -> Tuple[List[Message], str]:
    """
    Compact old messages into summary.

    Args:
        messages: List of messages to compact.
        keep_recent: Number of recent messages to keep.
        context_window: Maximum context window size (used for info).
        summary_fn: Optional async function to generate summary from messages.

    Returns:
        Tuple of (compacted_messages, summary_text).
    """
    to_compact, to_keep = select_compaction_range(messages, keep_recent)

    if not to_compact:
        return messages, ""

    # Generate summary (use provided function or placeholder)
    if summary_fn:
        summary_text = await summary_fn(to_compact)
    else:
        summary_text = _placeholder_summary(to_compact)

    # Create summary message
    summary_message = AssistantMessage(
        role="assistant",
        content=[{"type": "text", "text": f"[COMPACTED CONTEXT SUMMARY]\n{summary_text}"}],
        api="compaction",
        provider="internal",
        model="compaction",
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "total": 0.0},
        },
        stop_reason="stop",
        timestamp=int(datetime.now().timestamp() * 1000),
    )

    return [summary_message] + to_keep, summary_text


def _placeholder_summary(messages: List[Message]) -> str:
    """
    Generate placeholder summary of messages.

    This provides a basic summary when no LLM-based summary function is provided.

    Args:
        messages: Messages to summarize.

    Returns:
        Summary text.
    """
    summary_lines = []

    user_count = 0
    assistant_count = 0
    tool_result_count = 0
    tool_calls = []

    for msg in messages:
        role = msg.role
        if role == "user":
            user_count += 1
        elif role == "assistant":
            assistant_count += 1
            content = msg.content
            if isinstance(content, list):
                for block in content:
                    block_type = _get_block_type(block)
                    if block_type == "toolCall":
                        name, _ = _get_tool_call_info(block)
                        tool_calls.append(name)
        elif role == "toolResult":
            tool_result_count += 1

    summary_lines.append(
        f"Context contained {user_count} user messages, {assistant_count} assistant responses, "
        f"and {tool_result_count} tool results."
    )
    if tool_calls:
        unique_tools = sorted(set(tool_calls))
        summary_lines.append(f"Tool calls made: {', '.join(unique_tools)}")

    return "\n".join(summary_lines)


def extract_compaction_path(messages: List[Message], compaction_id: str) -> List[Message]:
    """
    Extract messages covered by a compaction entry.

    Walks backward from compaction_id, collecting messages until hitting
    another compaction or the start.

    Args:
        messages: List of all messages.
        compaction_id: ID of compaction entry to extract.

    Returns:
        List of messages covered by the compaction.
    """
    # Placeholder - would need entry tree to implement properly
    return messages


__all__ = [
    "estimate_context_tokens",
    "should_compact",
    "select_compaction_range",
    "compact_messages",
    "extract_compaction_path",
]