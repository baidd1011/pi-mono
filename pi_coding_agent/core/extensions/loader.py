"""
Extension loading and discovery.

Extensions are discovered in two locations:
- Local: <workspace>/.pi/extensions/*.py
- Global: ~/.pi/extensions/*.py

Each extension module must define an `extension_factory` function
that returns an Extension object.
"""

import importlib.util
from pathlib import Path
from typing import List, Optional

from .types import Extension


def discover_extensions(workspace_dir: str) -> List[str]:
    """Find all extension module paths.

    Searches both local (workspace) and global (home) extension directories.

    Args:
        workspace_dir: The workspace directory path.

    Returns:
        List of paths to extension Python files.
    """
    extensions = []

    # Local extensions: .pi/extensions/
    local_dir = Path(workspace_dir) / ".pi" / "extensions"
    if local_dir.exists():
        for f in local_dir.glob("*.py"):
            if f.name != "__init__.py":
                extensions.append(str(f))

    # Global extensions: ~/.pi/extensions/
    global_dir = Path.home() / ".pi" / "extensions"
    if global_dir.exists():
        for f in global_dir.glob("*.py"):
            if f.name != "__init__.py":
                extensions.append(str(f))

    return extensions


def load_extension(path: str) -> Optional[Extension]:
    """Load extension from Python file.

    The extension module must define an `extension_factory` function
    that returns an Extension object.

    Args:
        path: Path to the extension Python file.

    Returns:
        Extension object if loaded successfully, None otherwise.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "extension_module",
            path
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Extension must define extension_factory function
        if hasattr(module, "extension_factory"):
            ext = module.extension_factory()
            if isinstance(ext, Extension):
                return ext

        return None
    except Exception:
        return None


def load_all_extensions(workspace_dir: str) -> List[Extension]:
    """Load all discovered extensions.

    Args:
        workspace_dir: The workspace directory path.

    Returns:
        List of successfully loaded Extension objects.
    """
    paths = discover_extensions(workspace_dir)
    extensions = []

    for path in paths:
        ext = load_extension(path)
        if ext:
            extensions.append(ext)

    return extensions