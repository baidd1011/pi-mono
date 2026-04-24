"""
Extension types and definitions.

This module defines the data structures for extensions
that can extend pi-coding-agent with custom tools, commands,
and keybindings.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any


@dataclass
class ExtensionCommand:
    """Command contributed by extension.

    Commands are actions that can be triggered by keybindings
    or programmatically.

    Attributes:
        id: Unique command identifier (e.g., "myExtension.myCommand").
        title: Human-readable title for UI display.
        handler: Callable that executes the command.
    """

    id: str
    title: str
    handler: Callable


@dataclass
class ExtensionKeybinding:
    """Keybinding contributed by extension.

    Keybindings map keyboard shortcuts to commands.

    Attributes:
        key: Key combination (e.g., "ctrl+shift+f").
        command_id: ID of the command to execute.
        when: Optional condition for when keybinding is active.
    """

    key: str
    command_id: str
    when: Optional[str] = None


@dataclass
class Extension:
    """Extension definition.

    Extensions can contribute tools, commands, keybindings,
    and hook into the agent lifecycle.

    Attributes:
        name: Extension name.
        version: Extension version (semver recommended).
        description: Brief description of what the extension does.
        tools: List of tools to register.
        commands: List of commands contributed by this extension.
        keybindings: List of keybindings contributed by this extension.
        on_load: Called when extension is loaded.
        on_activate: Called when extension is activated.
        on_deactivate: Called when extension is deactivated.
    """

    name: str
    version: str = "0.1.0"
    description: str = ""

    # Extension capabilities
    tools: List[Any] = field(default_factory=list)
    commands: List[ExtensionCommand] = field(default_factory=list)
    keybindings: List[ExtensionKeybinding] = field(default_factory=list)

    # Lifecycle hooks
    on_load: Optional[Callable[[], None]] = None
    on_activate: Optional[Callable[[], None]] = None
    on_deactivate: Optional[Callable[[], None]] = None