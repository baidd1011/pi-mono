"""
Modes module - provides different interfaces for running the coding agent.

- print_mode: CLI single-shot output
- rpc: JSON-RPC over stdin/stdout for process integration
- sdk: Embedded in other Python applications
"""

from .print_mode import run_print_mode, print_and_exit
from .rpc import RPCHandler, run_rpc_mode
from .sdk import create_sdk_session, run_sdk_prompt


__all__ = [
    "run_print_mode",
    "print_and_exit",
    "RPCHandler",
    "run_rpc_mode",
    "create_sdk_session",
    "run_sdk_prompt",
]