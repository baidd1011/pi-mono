# tests/test_extensions.py
"""Tests for pi-coding-agent extension system."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestExtensionTypes:
    """Tests for extension type definitions."""

    def test_extension_class_exists(self):
        """Extension class is importable."""
        from pi_coding_agent.core.extensions.types import Extension

        assert Extension is not None

    def test_extension_command_class_exists(self):
        """ExtensionCommand class is importable."""
        from pi_coding_agent.core.extensions.types import ExtensionCommand

        assert ExtensionCommand is not None

    def test_extension_keybinding_class_exists(self):
        """ExtensionKeybinding class is importable."""
        from pi_coding_agent.core.extensions.types import ExtensionKeybinding

        assert ExtensionKeybinding is not None

    def test_extension_creation(self):
        """Extension can be created with name."""
        from pi_coding_agent.core.extensions.types import Extension

        ext = Extension(name="test-extension")

        assert ext.name == "test-extension"
        assert ext.version == "0.1.0"
        assert ext.description == ""
        assert ext.tools == []
        assert ext.commands == []
        assert ext.keybindings == []

    def test_extension_with_all_fields(self):
        """Extension can be created with all fields."""
        from pi_coding_agent.core.extensions.types import (
            Extension,
            ExtensionCommand,
            ExtensionKeybinding,
        )

        def on_load():
            pass

        def handler():
            pass

        ext = Extension(
            name="full-extension",
            version="1.0.0",
            description="A full extension",
            tools=[MagicMock()],
            commands=[ExtensionCommand(id="cmd1", title="Command 1", handler=handler)],
            keybindings=[ExtensionKeybinding(key="ctrl+f", command_id="cmd1")],
            on_load=on_load,
        )

        assert ext.name == "full-extension"
        assert ext.version == "1.0.0"
        assert ext.description == "A full extension"
        assert len(ext.tools) == 1
        assert len(ext.commands) == 1
        assert len(ext.keybindings) == 1
        assert ext.on_load == on_load

    def test_extension_command_creation(self):
        """ExtensionCommand can be created."""
        from pi_coding_agent.core.extensions.types import ExtensionCommand

        def handler():
            return "result"

        cmd = ExtensionCommand(id="test.cmd", title="Test Command", handler=handler)

        assert cmd.id == "test.cmd"
        assert cmd.title == "Test Command"
        assert cmd.handler == handler
        assert cmd.handler() == "result"

    def test_extension_keybinding_creation(self):
        """ExtensionKeybinding can be created."""
        from pi_coding_agent.core.extensions.types import ExtensionKeybinding

        kb = ExtensionKeybinding(key="ctrl+shift+f", command_id="test.cmd")

        assert kb.key == "ctrl+shift+f"
        assert kb.command_id == "test.cmd"
        assert kb.when is None

    def test_extension_keybinding_with_when(self):
        """ExtensionKeybinding can have when condition."""
        from pi_coding_agent.core.extensions.types import ExtensionKeybinding

        kb = ExtensionKeybinding(
            key="ctrl+s", command_id="save", when="editorFocus"
        )

        assert kb.key == "ctrl+s"
        assert kb.command_id == "save"
        assert kb.when == "editorFocus"


class TestExtensionDiscovery:
    """Tests for extension discovery."""

    def test_discover_extensions_function_exists(self):
        """discover_extensions function is importable."""
        from pi_coding_agent.core.extensions.loader import discover_extensions

        assert callable(discover_extensions)

    def test_discover_extensions_empty_workspace(self):
        """discover_extensions returns empty list for workspace without extensions."""
        from pi_coding_agent.core.extensions.loader import discover_extensions

        with tempfile.TemporaryDirectory() as workspace:
            paths = discover_extensions(workspace)
            assert paths == []

    def test_discover_extensions_local(self):
        """discover_extensions finds local extensions."""
        from pi_coding_agent.core.extensions.loader import discover_extensions

        with tempfile.TemporaryDirectory() as workspace:
            # Create .pi/extensions directory
            ext_dir = Path(workspace) / ".pi" / "extensions"
            ext_dir.mkdir(parents=True)

            # Create an extension file
            ext_file = ext_dir / "my_ext.py"
            ext_file.write_text("# Extension file\n")

            paths = discover_extensions(workspace)

            assert len(paths) == 1
            assert "my_ext.py" in paths[0]

    def test_discover_extensions_skips_init(self):
        """discover_extensions skips __init__.py files."""
        from pi_coding_agent.core.extensions.loader import discover_extensions

        with tempfile.TemporaryDirectory() as workspace:
            ext_dir = Path(workspace) / ".pi" / "extensions"
            ext_dir.mkdir(parents=True)

            # Create __init__.py and regular extension
            init_file = ext_dir / "__init__.py"
            init_file.write_text("# Init\n")
            ext_file = ext_dir / "real_ext.py"
            ext_file.write_text("# Real extension\n")

            paths = discover_extensions(workspace)

            assert len(paths) == 1
            assert "real_ext.py" in paths[0]

    def test_discover_extensions_multiple_files(self):
        """discover_extensions finds multiple extension files."""
        from pi_coding_agent.core.extensions.loader import discover_extensions

        with tempfile.TemporaryDirectory() as workspace:
            ext_dir = Path(workspace) / ".pi" / "extensions"
            ext_dir.mkdir(parents=True)

            # Create multiple extension files
            for name in ["ext1.py", "ext2.py", "ext3.py"]:
                ext_file = ext_dir / name
                ext_file.write_text(f"# {name}\n")

            paths = discover_extensions(workspace)

            assert len(paths) == 3

    def test_discover_extensions_global(self):
        """discover_extensions finds global extensions in ~/.pi/extensions."""
        from pi_coding_agent.core.extensions.loader import discover_extensions

        with tempfile.TemporaryDirectory() as workspace:
            # Mock the home directory
            with patch("pi_coding_agent.core.extensions.loader.Path.home") as mock_home:
                mock_home.return_value = Path(workspace)

                # Create global extensions directory (in temp workspace as mock home)
                global_dir = Path(workspace) / ".pi" / "extensions"
                global_dir.mkdir(parents=True)

                ext_file = global_dir / "global_ext.py"
                ext_file.write_text("# Global extension\n")

                # Use a different directory as workspace
                with tempfile.TemporaryDirectory() as other_workspace:
                    paths = discover_extensions(other_workspace)

                    # Should find the global extension
                    assert len(paths) == 1
                    assert "global_ext.py" in paths[0]


class TestExtensionLoading:
    """Tests for extension loading."""

    def test_load_extension_function_exists(self):
        """load_extension function is importable."""
        from pi_coding_agent.core.extensions.loader import load_extension

        assert callable(load_extension)

    def test_load_extension_returns_none_for_invalid(self):
        """load_extension returns None for invalid file."""
        from pi_coding_agent.core.extensions.loader import load_extension

        with tempfile.TemporaryDirectory() as tmp:
            invalid_file = Path(tmp) / "invalid.py"
            invalid_file.write_text("syntax error here {{{\n")

            result = load_extension(str(invalid_file))
            assert result is None

    def test_load_extension_returns_none_without_factory(self):
        """load_extension returns None if no extension_factory."""
        from pi_coding_agent.core.extensions.loader import load_extension

        with tempfile.TemporaryDirectory() as tmp:
            no_factory_file = Path(tmp) / "no_factory.py"
            no_factory_file.write_text("# No factory function\n")

            result = load_extension(str(no_factory_file))
            assert result is None

    def test_load_extension_with_factory(self):
        """load_extension loads extension with extension_factory."""
        from pi_coding_agent.core.extensions.loader import load_extension
        from pi_coding_agent.core.extensions.types import Extension

        with tempfile.TemporaryDirectory() as tmp:
            ext_file = Path(tmp) / "my_extension.py"
            ext_file.write_text("""
from pi_coding_agent.core.extensions.types import Extension

def extension_factory():
    return Extension(name="my-extension", version="1.0.0")
""")

            result = load_extension(str(ext_file))

            assert result is not None
            assert isinstance(result, Extension)
            assert result.name == "my-extension"
            assert result.version == "1.0.0"

    def test_load_extension_factory_returns_wrong_type(self):
        """load_extension returns None if factory returns non-Extension."""
        from pi_coding_agent.core.extensions.loader import load_extension

        with tempfile.TemporaryDirectory() as tmp:
            ext_file = Path(tmp) / "wrong_type.py"
            ext_file.write_text("""
def extension_factory():
    return "not an extension"
""")

            result = load_extension(str(ext_file))
            assert result is None

    def test_load_extension_with_full_extension(self):
        """load_extension loads full extension with tools and commands."""
        from pi_coding_agent.core.extensions.loader import load_extension
        from pi_coding_agent.core.extensions.types import Extension

        with tempfile.TemporaryDirectory() as tmp:
            ext_file = Path(tmp) / "full_extension.py"
            ext_file.write_text("""
from pi_coding_agent.core.extensions.types import Extension, ExtensionCommand

def my_handler():
    print("Hello")

def extension_factory():
    return Extension(
        name="full-extension",
        version="2.0.0",
        description="Full extension with commands",
        commands=[
            ExtensionCommand(id="cmd.hello", title="Say Hello", handler=my_handler)
        ]
    )
""")

            result = load_extension(str(ext_file))

            assert result is not None
            assert result.name == "full-extension"
            assert result.version == "2.0.0"
            assert result.description == "Full extension with commands"
            assert len(result.commands) == 1
            assert result.commands[0].id == "cmd.hello"


class TestLoadAllExtensions:
    """Tests for loading all extensions."""

    def test_load_all_extensions_function_exists(self):
        """load_all_extensions function is importable."""
        from pi_coding_agent.core.extensions.loader import load_all_extensions

        assert callable(load_all_extensions)

    def test_load_all_extensions_empty(self):
        """load_all_extensions returns empty list for workspace without extensions."""
        from pi_coding_agent.core.extensions.loader import load_all_extensions

        with tempfile.TemporaryDirectory() as workspace:
            extensions = load_all_extensions(workspace)
            assert extensions == []

    def test_load_all_extensions_with_valid(self):
        """load_all_extensions loads valid extensions."""
        from pi_coding_agent.core.extensions.loader import load_all_extensions
        from pi_coding_agent.core.extensions.types import Extension

        with tempfile.TemporaryDirectory() as workspace:
            ext_dir = Path(workspace) / ".pi" / "extensions"
            ext_dir.mkdir(parents=True)

            # Create valid extension
            ext_file = ext_dir / "valid_ext.py"
            ext_file.write_text("""
from pi_coding_agent.core.extensions.types import Extension

def extension_factory():
    return Extension(name="valid-extension")
""")

            extensions = load_all_extensions(workspace)

            assert len(extensions) == 1
            assert isinstance(extensions[0], Extension)
            assert extensions[0].name == "valid-extension"

    def test_load_all_extensions_mixed(self):
        """load_all_extensions loads only valid extensions, skipping invalid."""
        from pi_coding_agent.core.extensions.loader import load_all_extensions
        from pi_coding_agent.core.extensions.types import Extension

        with tempfile.TemporaryDirectory() as workspace:
            ext_dir = Path(workspace) / ".pi" / "extensions"
            ext_dir.mkdir(parents=True)

            # Valid extension
            valid_file = ext_dir / "valid.py"
            valid_file.write_text("""
from pi_coding_agent.core.extensions.types import Extension
def extension_factory():
    return Extension(name="valid")
""")

            # Invalid extension (no factory)
            invalid_file = ext_dir / "invalid.py"
            invalid_file.write_text("# No factory\n")

            # Broken extension (syntax error)
            broken_file = ext_dir / "broken.py"
            broken_file.write_text("syntax error {{{\n")

            extensions = load_all_extensions(workspace)

            assert len(extensions) == 1
            assert extensions[0].name == "valid"


class TestExtensionsModuleExports:
    """Tests for extensions module exports."""

    def test_import_from_package(self):
        """Extensions can be imported from pi_coding_agent."""
        from pi_coding_agent import (
            Extension,
            ExtensionCommand,
            ExtensionKeybinding,
            discover_extensions,
            load_extension,
            load_all_extensions,
        )

        assert Extension is not None
        assert ExtensionCommand is not None
        assert ExtensionKeybinding is not None
        assert callable(discover_extensions)
        assert callable(load_extension)
        assert callable(load_all_extensions)

    def test_import_from_extensions_module(self):
        """Extensions can be imported from extensions submodule."""
        from pi_coding_agent.core.extensions import (
            Extension,
            ExtensionCommand,
            ExtensionKeybinding,
            discover_extensions,
            load_extension,
            load_all_extensions,
        )

        assert Extension is not None
        assert ExtensionCommand is not None
        assert ExtensionKeybinding is not None
        assert callable(discover_extensions)
        assert callable(load_extension)
        assert callable(load_all_extensions)