# tests/test_tools.py
"""Tests for all 7 coding tools."""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from pi_coding_agent.core.tools import (
    create_coding_tools,
    create_read_tool,
    create_write_tool,
    create_edit_tool,
    create_bash_tool,
    create_grep_tool,
    create_find_tool,
    create_ls_tool,
)
from pi_agent_core import AgentTool
from pi_ai import TextContent


class TestCreateCodingTools:
    """Tests for create_coding_tools function."""

    def test_creates_all_7_tools(self):
        """create_coding_tools returns exactly 7 tools."""
        tools = create_coding_tools("/tmp")
        assert len(tools) == 7

    def test_all_tools_have_correct_names(self):
        """All tools have the expected names."""
        tools = create_coding_tools("/tmp")
        expected_names = ["read", "write", "edit", "bash", "grep", "find", "ls"]
        actual_names = [t.name for t in tools]
        assert sorted(actual_names) == sorted(expected_names)

    def test_all_tools_are_agent_tools(self):
        """All returned tools are AgentTool instances."""
        tools = create_coding_tools("/tmp")
        for tool in tools:
            assert isinstance(tool, AgentTool)
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'label')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'parameters')
            assert hasattr(tool, 'execute')


class TestReadTool:
    """Tests for read tool."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def read_tool(self, temp_dir):
        """Create a read tool with temp directory."""
        return create_read_tool(temp_dir)

    def test_read_tool_creation(self, read_tool):
        """Read tool is created correctly."""
        assert read_tool.name == "read"
        assert read_tool.label == "Read File"
        assert "path" in read_tool.parameters.get("properties", {})

    @pytest.mark.asyncio
    async def test_read_missing_path(self, read_tool):
        """Read tool returns error for missing path."""
        result = await read_tool.execute("test-id", {}, None, None)
        assert result.content[0].text.startswith("Error:")
        assert "path" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, read_tool, temp_dir):
        """Read tool returns error for nonexistent file."""
        result = await read_tool.execute("test-id", {"path": "nonexistent.txt"}, None, None)
        assert "Error" in result.content[0].text
        assert "not found" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_read_existing_file(self, read_tool, temp_dir):
        """Read tool can read an existing file."""
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Hello, World!\nLine 2\nLine 3")

        result = await read_tool.execute("test-id", {"path": "test.txt"}, None, None)

        assert result.content[0].text.startswith("1\t")
        assert "Hello, World!" in result.content[0].text
        assert "Line 2" in result.content[0].text

    @pytest.mark.asyncio
    async def test_read_with_offset(self, read_tool, temp_dir):
        """Read tool supports offset parameter."""
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")

        result = await read_tool.execute("test-id", {"path": "test.txt", "offset": 2}, None, None)

        # Should start from line 2 (1-indexed)
        assert "2\t" in result.content[0].text
        assert "Line 2" in result.content[0].text
        assert "Line 1" not in result.content[0].text

    @pytest.mark.asyncio
    async def test_read_with_limit(self, read_tool, temp_dir):
        """Read tool supports limit parameter."""
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")

        result = await read_tool.execute("test-id", {"path": "test.txt", "limit": 2}, None, None)

        # Should only show 2 lines
        assert "Line 1" in result.content[0].text
        assert "Line 2" in result.content[0].text
        assert "Line 3" not in result.content[0].text

    @pytest.mark.asyncio
    async def test_read_path_outside_workspace(self, read_tool, temp_dir):
        """Read tool rejects paths outside workspace."""
        result = await read_tool.execute("test-id", {"path": "../outside.txt"}, None, None)
        assert "Error" in result.content[0].text
        assert "outside" in result.content[0].text.lower()


class TestWriteTool:
    """Tests for write tool."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def write_tool(self, temp_dir):
        """Create a write tool with temp directory."""
        return create_write_tool(temp_dir)

    def test_write_tool_creation(self, write_tool):
        """Write tool is created correctly."""
        assert write_tool.name == "write"
        assert write_tool.label == "Write File"

    @pytest.mark.asyncio
    async def test_write_missing_path(self, write_tool):
        """Write tool returns error for missing path."""
        result = await write_tool.execute("test-id", {"content": "test"}, None, None)
        assert "Error" in result.content[0].text
        assert "path" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_write_missing_content(self, write_tool):
        """Write tool returns error for missing content."""
        result = await write_tool.execute("test-id", {"path": "test.txt"}, None, None)
        assert "Error" in result.content[0].text
        assert "content" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_write_creates_file(self, write_tool, temp_dir):
        """Write tool creates a new file."""
        result = await write_tool.execute(
            "test-id",
            {"path": "new.txt", "content": "Hello, World!"},
            None, None
        )

        assert "Successfully" in result.content[0].text
        assert Path(temp_dir, "new.txt").exists()
        assert Path(temp_dir, "new.txt").read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_write_creates_parent_directories(self, write_tool, temp_dir):
        """Write tool creates parent directories."""
        result = await write_tool.execute(
            "test-id",
            {"path": "subdir/deep/new.txt", "content": "nested"},
            None, None
        )

        assert "Successfully" in result.content[0].text
        assert Path(temp_dir, "subdir/deep/new.txt").exists()

    @pytest.mark.asyncio
    async def test_write_overwrites_file(self, write_tool, temp_dir):
        """Write tool overwrites existing file."""
        test_file = Path(temp_dir) / "existing.txt"
        test_file.write_text("Old content")

        result = await write_tool.execute(
            "test-id",
            {"path": "existing.txt", "content": "New content"},
            None, None
        )

        assert "Successfully" in result.content[0].text
        assert test_file.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_write_path_outside_workspace(self, write_tool):
        """Write tool rejects paths outside workspace."""
        result = await write_tool.execute(
            "test-id",
            {"path": "../outside.txt", "content": "test"},
            None, None
        )
        assert "Error" in result.content[0].text
        assert "outside" in result.content[0].text.lower()


class TestEditTool:
    """Tests for edit tool."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def edit_tool(self, temp_dir):
        """Create an edit tool with temp directory."""
        return create_edit_tool(temp_dir)

    def test_edit_tool_creation(self, edit_tool):
        """Edit tool is created correctly."""
        assert edit_tool.name == "edit"
        assert edit_tool.label == "Edit File"

    @pytest.mark.asyncio
    async def test_edit_missing_path(self, edit_tool):
        """Edit tool returns error for missing path."""
        result = await edit_tool.execute(
            "test-id",
            {"old_string": "old", "new_string": "new"},
            None, None
        )
        assert "Error" in result.content[0].text
        assert "path" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self, edit_tool, temp_dir):
        """Edit tool returns error for nonexistent file."""
        result = await edit_tool.execute(
            "test-id",
            {"path": "nonexistent.txt", "old_string": "old", "new_string": "new"},
            None, None
        )
        assert "Error" in result.content[0].text
        assert "not found" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_edit_single_replacement(self, edit_tool, temp_dir):
        """Edit tool performs single string replacement."""
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Hello, World!\nGoodbye, World!")

        result = await edit_tool.execute(
            "test-id",
            {"path": "test.txt", "old_string": "World", "new_string": "Python"},
            None, None
        )

        # Should fail because "World" appears twice (not unique)
        assert "Error" in result.content[0].text
        assert "unique" in result.content[0].text.lower() or "occurrences" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_edit_unique_replacement(self, edit_tool, temp_dir):
        """Edit tool performs unique string replacement."""
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Hello, World!\nGoodbye, Universe!")

        result = await edit_tool.execute(
            "test-id",
            {"path": "test.txt", "old_string": "World", "new_string": "Python"},
            None, None
        )

        assert "Successfully" in result.content[0].text
        assert test_file.read_text() == "Hello, Python!\nGoodbye, Universe!"

    @pytest.mark.asyncio
    async def test_edit_not_found(self, edit_tool, temp_dir):
        """Edit tool returns error when old_string not found."""
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Hello, World!")

        result = await edit_tool.execute(
            "test-id",
            {"path": "test.txt", "old_string": "NotInFile", "new_string": "New"},
            None, None
        )

        assert "Error" in result.content[0].text
        assert "not find" in result.content[0].text.lower() or "could not" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_edit_with_replace_all(self, edit_tool, temp_dir):
        """Edit tool supports replace_all for multiple occurrences."""
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("World, World, World")

        result = await edit_tool.execute(
            "test-id",
            {"path": "test.txt", "old_string": "World", "new_string": "Python", "replace_all": True},
            None, None
        )

        assert "Successfully" in result.content[0].text
        assert test_file.read_text() == "Python, Python, Python"


class TestBashTool:
    """Tests for bash tool."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def bash_tool(self, temp_dir):
        """Create a bash tool with temp directory."""
        return create_bash_tool(temp_dir)

    def test_bash_tool_creation(self, bash_tool):
        """Bash tool is created correctly."""
        assert bash_tool.name == "bash"
        assert bash_tool.label == "Run Command"

    @pytest.mark.asyncio
    async def test_bash_missing_command(self, bash_tool):
        """Bash tool returns error for missing command."""
        result = await bash_tool.execute("test-id", {}, None, None)
        assert "Error" in result.content[0].text
        assert "command" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_bash_simple_command(self, bash_tool, temp_dir):
        """Bash tool can run a simple command."""
        # Use echo command (works on both Windows and Unix)
        import sys
        if sys.platform == "win32":
            cmd = "echo hello"
        else:
            cmd = "echo hello"

        result = await bash_tool.execute("test-id", {"command": cmd}, None, None)

        # Check for output
        assert "hello" in result.content[0].text.lower() or result.details.exit_code == 0

    @pytest.mark.asyncio
    async def test_bash_timeout(self, bash_tool):
        """Bash tool respects timeout."""
        import sys
        if sys.platform == "win32":
            # Windows: use ping to simulate a long-running command
            cmd = "ping -n 10 127.0.0.1"
        else:
            # Unix: sleep for 10 seconds
            cmd = "sleep 10"

        result = await bash_tool.execute(
            "test-id",
            {"command": cmd, "timeout": 1},  # 1 second timeout
            None, None
        )

        # Should have timed out
        assert "timeout" in result.content[0].text.lower() or "timed out" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_bash_exit_code(self, bash_tool):
        """Bash tool reports non-zero exit codes."""
        import sys
        if sys.platform == "win32":
            # Windows: use a command that will fail
            cmd = "exit 1"
        else:
            cmd = "false"

        result = await bash_tool.execute("test-id", {"command": cmd}, None, None)

        # Should report exit code
        if result.details.exit_code is not None:
            assert result.details.exit_code != 0


class TestGrepTool:
    """Tests for grep tool."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            # Create some test files
            Path(d, "file1.txt").write_text("Hello World\nPython is great\n")
            Path(d, "file2.txt").write_text("Goodbye World\nJavaScript is okay\n")
            Path(d, "subdir").mkdir()
            Path(d, "subdir", "file3.txt").write_text("Hello Python\n")
            yield d

    @pytest.fixture
    def grep_tool(self, temp_dir):
        """Create a grep tool with temp directory."""
        return create_grep_tool(temp_dir)

    def test_grep_tool_creation(self, grep_tool):
        """Grep tool is created correctly."""
        assert grep_tool.name == "grep"
        assert grep_tool.label == "Search Content"

    @pytest.mark.asyncio
    async def test_grep_missing_pattern(self, grep_tool):
        """Grep tool returns error for missing pattern."""
        result = await grep_tool.execute("test-id", {}, None, None)
        assert "Error" in result.content[0].text
        assert "pattern" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_grep_finds_matches(self, grep_tool, temp_dir):
        """Grep tool finds matching lines."""
        result = await grep_tool.execute("test-id", {"pattern": "Hello"}, None, None)

        assert "Hello" in result.content[0].text
        assert "file1.txt" in result.content[0].text or "file3.txt" in result.content[0].text

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, grep_tool, temp_dir):
        """Grep tool returns message for no matches."""
        result = await grep_tool.execute("test-id", {"pattern": "NotInFiles"}, None, None)

        assert "No matches" in result.content[0].text

    @pytest.mark.asyncio
    async def test_grep_case_insensitive(self, grep_tool, temp_dir):
        """Grep tool supports case-insensitive search."""
        result = await grep_tool.execute(
            "test-id",
            {"pattern": "hello", "-i": True},
            None, None
        )

        assert "Hello" in result.content[0].text or "hello" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_grep_with_glob(self, grep_tool, temp_dir):
        """Grep tool supports glob pattern filtering."""
        result = await grep_tool.execute(
            "test-id",
            {"pattern": "World", "glob": "*.txt"},
            None, None
        )

        # Should find "World" in .txt files
        assert "World" in result.content[0].text


class TestFindTool:
    """Tests for find tool."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            # Create some test files
            Path(d, "file1.txt").touch()
            Path(d, "file2.py").touch()
            Path(d, "subdir").mkdir()
            Path(d, "subdir", "file3.txt").touch()
            Path(d, "subdir", "file4.py").touch()
            yield d

    @pytest.fixture
    def find_tool(self, temp_dir):
        """Create a find tool with temp directory."""
        return create_find_tool(temp_dir)

    def test_find_tool_creation(self, find_tool):
        """Find tool is created correctly."""
        assert find_tool.name == "find"
        assert find_tool.label == "Find Files"

    @pytest.mark.asyncio
    async def test_find_missing_pattern(self, find_tool):
        """Find tool returns error for missing pattern."""
        result = await find_tool.execute("test-id", {}, None, None)
        assert "Error" in result.content[0].text
        assert "pattern" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_find_by_extension(self, find_tool, temp_dir):
        """Find tool can find files by extension."""
        result = await find_tool.execute("test-id", {"pattern": "*.py"}, None, None)

        assert "file2.py" in result.content[0].text
        assert "file4.py" in result.content[0].text
        assert ".txt" not in result.content[0].text  # Should not include .txt files

    @pytest.mark.asyncio
    async def test_find_no_matches(self, find_tool, temp_dir):
        """Find tool returns message for no matches."""
        result = await find_tool.execute("test-id", {"pattern": "*.md"}, None, None)

        assert "No files" in result.content[0].text

    @pytest.mark.asyncio
    async def test_find_recursive(self, find_tool, temp_dir):
        """Find tool searches recursively."""
        result = await find_tool.execute("test-id", {"pattern": "*.txt"}, None, None)

        assert "file1.txt" in result.content[0].text
        assert "subdir/file3.txt" in result.content[0].text.replace('\\', '/')


class TestLsTool:
    """Tests for ls tool."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            # Create some test entries
            Path(d, "file1.txt").touch()
            Path(d, "file2.py").touch()
            Path(d, "subdir").mkdir()
            Path(d, "emptydir").mkdir()
            yield d

    @pytest.fixture
    def ls_tool(self, temp_dir):
        """Create a ls tool with temp directory."""
        return create_ls_tool(temp_dir)

    def test_ls_tool_creation(self, ls_tool):
        """Ls tool is created correctly."""
        assert ls_tool.name == "ls"
        assert ls_tool.label == "List Directory"

    @pytest.mark.asyncio
    async def test_ls_default_directory(self, ls_tool, temp_dir):
        """Ls tool lists current directory by default."""
        result = await ls_tool.execute("test-id", {}, None, None)

        assert "file1.txt" in result.content[0].text
        assert "file2.py" in result.content[0].text
        assert "subdir/" in result.content[0].text

    @pytest.mark.asyncio
    async def test_ls_directory_suffix(self, ls_tool, temp_dir):
        """Ls tool adds '/' suffix to directories."""
        result = await ls_tool.execute("test-id", {}, None, None)

        # Check that directories have suffix
        assert "subdir/" in result.content[0].text
        # Check that files don't have suffix
        assert "file1.txt/" not in result.content[0].text

    @pytest.mark.asyncio
    async def test_ls_empty_directory(self, ls_tool, temp_dir):
        """Ls tool handles empty directory."""
        result = await ls_tool.execute("test-id", {"path": "emptydir"}, None, None)

        assert "empty" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_ls_nonexistent_path(self, ls_tool):
        """Ls tool returns error for nonexistent path."""
        result = await ls_tool.execute("test-id", {"path": "nonexistent"}, None, None)

        assert "Error" in result.content[0].text
        assert "not found" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_ls_file_not_directory(self, ls_tool, temp_dir):
        """Ls tool returns error when path is a file."""
        result = await ls_tool.execute("test-id", {"path": "file1.txt"}, None, None)

        assert "Error" in result.content[0].text
        assert "Not a directory" in result.content[0].text


class TestPathSecurity:
    """Tests for path security across all tools."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def all_tools(self, temp_dir):
        """Create all tools with temp directory."""
        return {
            "read": create_read_tool(temp_dir),
            "write": create_write_tool(temp_dir),
            "edit": create_edit_tool(temp_dir),
            "bash": create_bash_tool(temp_dir),
            "grep": create_grep_tool(temp_dir),
            "find": create_find_tool(temp_dir),
            "ls": create_ls_tool(temp_dir),
        }

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, all_tools):
        """All tools block path traversal outside workspace."""
        # Test read
        result = await all_tools["read"].execute("test-id", {"path": "../etc/passwd"}, None, None)
        assert "Error" in result.content[0].text

        # Test write
        result = await all_tools["write"].execute("test-id", {"path": "../outside.txt", "content": "test"}, None, None)
        assert "Error" in result.content[0].text

        # Test ls
        result = await all_tools["ls"].execute("test-id", {"path": ".."}, None, None)
        assert "Error" in result.content[0].text

    @pytest.mark.asyncio
    async def test_absolute_path_outside_blocked(self, all_tools):
        """All tools block absolute paths outside workspace."""
        # Test read with absolute path outside workspace
        result = await all_tools["read"].execute("test-id", {"path": "/etc/passwd"}, None, None)
        assert "Error" in result.content[0].text


class TestToolParameters:
    """Tests for tool parameter schemas."""

    def test_all_tools_have_valid_parameters(self):
        """All tools have valid JSON Schema parameters."""
        tools = create_coding_tools("/tmp")

        for tool in tools:
            params = tool.parameters
            assert params is not None
            assert params.get("type") == "object"
            assert "properties" in params

            # Required fields should be in properties
            required = params.get("required", [])
            for req in required:
                assert req in params["properties"]