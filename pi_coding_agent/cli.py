"""
CLI entry point for pi-coding-agent.

Provides command-line interface with multiple modes:
- Interactive TUI mode (default)
- Print mode (single-shot output)
- RPC mode (JSON-RPC over stdin/stdout)
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pi Coding Agent - Interactive coding assistant",
        prog="pi",
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Initial prompt (starts interactive mode if omitted)",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="Workspace directory",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Model ID to use",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "print", "rpc"],
        default="interactive",
        help="Run mode (default: interactive)",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable TUI (use basic input)",
    )
    parser.add_argument(
        "--session",
        "-s",
        help="Session ID to continue",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()

    if args.mode == "interactive" and not args.no_tui and not args.prompt:
        # Interactive TUI mode
        _run_interactive_tui(str(workspace), args.model, args.session)

    elif args.mode == "print" or (args.prompt and not args.no_tui):
        # Print mode with prompt
        from .modes.print_mode import print_and_exit

        asyncio.run(print_and_exit(args.prompt, str(workspace), args.model))

    elif args.mode == "rpc":
        # RPC mode
        from .modes.rpc import run_rpc_mode

        asyncio.run(run_rpc_mode(str(workspace)))

    else:
        # Basic interactive without TUI
        asyncio.run(_run_basic_interactive(workspace, args.model))


def _run_interactive_tui(
    workspace: str,
    model: Optional[str],
    session_id: Optional[str],
) -> None:
    """Run interactive TUI mode.

    Args:
        workspace: Workspace directory path.
        model: Optional model ID to use.
        session_id: Optional session ID to continue.
    """
    from .tui.app import PiCodingAgentApp
    from .core.agent_session import AgentSession

    session = AgentSession(workspace)

    # TODO: Load session if session_id provided
    # if session_id:
    #     session.load_session(session_id)

    app = PiCodingAgentApp(session)
    app.run()


async def _run_basic_interactive(
    workspace: Path,
    model: Optional[str],
) -> None:
    """Run basic interactive mode without TUI.

    Args:
        workspace: Workspace directory path.
        model: Optional model ID to use.
    """
    from .modes.print_mode import run_print_mode

    print("Pi Coding Agent (basic mode). Type 'exit' to quit.")
    print(f"Workspace: {workspace}")

    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() == "exit":
                break

            if not prompt.strip():
                continue

            output = await run_print_mode(prompt, str(workspace), model)
            print(f"\nAssistant: {output}")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()