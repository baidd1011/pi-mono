"""
Extension types and definitions.

Extensions allow extending pi-coding-agent with:
- Custom tools
- Commands
- Keybindings
- Lifecycle hooks
"""

from .types import Extension, ExtensionCommand, ExtensionKeybinding
from .loader import discover_extensions, load_extension, load_all_extensions


__all__ = [
    "Extension",
    "ExtensionCommand",
    "ExtensionKeybinding",
    "discover_extensions",
    "load_extension",
    "load_all_extensions",
]