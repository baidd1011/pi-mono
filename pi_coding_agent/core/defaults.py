"""
Default configuration values for pi-coding-agent.

This module defines the default model, thinking level, system prompt,
and other configuration constants used by the coding agent.
"""

from pi_ai import Model, Usage


# -----------------------------------------------------------------------------
# Default Model Configuration
# -----------------------------------------------------------------------------

DEFAULT_MODEL = Model(
    id="qwen-plus-latest",
    name="Qwen Plus",
    api="bailian-chat",
    provider="bailian",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    reasoning=False,
    input=["text", "image"],
    cost=Usage.Cost(input=0.8, output=2.0, cache_read=0.0, cache_write=0.0, total=2.8),
    context_window=128000,
    max_tokens=8192,
)


# -----------------------------------------------------------------------------
# Thinking Level
# -----------------------------------------------------------------------------

DEFAULT_THINKING_LEVEL = "off"


# -----------------------------------------------------------------------------
# System Prompt
# -----------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are a coding assistant that helps users write, edit, and analyze code.
You have access to tools for reading, writing, editing files, and running bash commands.
Use these tools to accomplish tasks efficiently and accurately.

When working on code:
- Read files to understand context before making changes
- Make precise edits rather than rewriting entire files
- Run tests to verify your changes work correctly
- Explain your reasoning clearly

Be thorough but efficient. Ask clarifying questions when needed."""


# -----------------------------------------------------------------------------
# Limits
# -----------------------------------------------------------------------------

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 100000


# -----------------------------------------------------------------------------
# Session Configuration
# -----------------------------------------------------------------------------

DEFAULT_SESSION_DIR_NAME = ".pi"
DEFAULT_SESSIONS_SUBDIR = "sessions"