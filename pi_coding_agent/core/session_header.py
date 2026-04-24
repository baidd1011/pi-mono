"""
Session file header types for pi-coding-agent.

Session files use JSONL (JSON Lines) format for efficient append-only storage.
Each file starts with a header containing metadata, followed by entries forming
a tree structure with id/parentId references.

Migration support ensures backward compatibility with older session versions.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Union
import json
import time


class SessionVersion(Enum):
    """Session file format version."""
    V1 = 1  # Original format without tree structure
    V2 = 2  # Added id/parentId for tree structure
    V3 = 3  # Renamed hookMessage role to custom


CURRENT_SESSION_VERSION = SessionVersion.V3


@dataclass
class SessionHeader:
    """Header at start of session file.

    Contains metadata about the session including version, ID, model info,
    and the current leaf position for tree traversal.
    """
    version: SessionVersion
    session_id: str
    model_id: str
    provider: str
    created_at: int  # Unix timestamp in milliseconds
    updated_at: int  # Unix timestamp in milliseconds
    cwd: str = ""  # Working directory where session was started
    leaf_entry_id: Optional[str] = None  # Current position in tree
    compaction_entries: List[str] = field(default_factory=list)  # Compaction entry IDs
    parent_session: Optional[str] = None  # Path to parent session if forked

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["type"] = "session"
        d["version"] = self.version.value
        # Use 'id' field (TypeScript compatible) instead of 'session_id'
        d["id"] = self.session_id
        d.pop("session_id", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionHeader":
        """Create from dictionary (parsed from JSON)."""
        version = SessionVersion(d.get("version", 1))
        return cls(
            version=version,
            session_id=d.get("id", d.get("session_id", "")),
            model_id=d.get("model_id", ""),
            provider=d.get("provider", ""),
            created_at=d.get("created_at", int(time.time() * 1000)),
            updated_at=d.get("updated_at", int(time.time() * 1000)),
            cwd=d.get("cwd", ""),
            leaf_entry_id=d.get("leaf_entry_id"),
            compaction_entries=d.get("compaction_entries", []),
            parent_session=d.get("parent_session"),
        )


@dataclass
class SessionEntryBase:
    """Base class for all session entries.

    Every entry has an id and parentId forming a tree structure.
    The parentId is null for root entries (first message in conversation).
    """
    type: str
    id: str
    parent_id: Optional[str]
    timestamp: str  # ISO 8601 format

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SessionMessageEntry(SessionEntryBase):
    """Entry containing an agent message.

    Stores the full message object for reconstruction on load.
    """
    message: Dict[str, Any]

    def __init__(
        self,
        id: str,
        parent_id: Optional[str],
        timestamp: str,
        message: Dict[str, Any],
    ):
        super().__init__(
            type="message",
            id=id,
            parent_id=parent_id,
            timestamp=timestamp,
        )
        self.message = message


@dataclass
class ThinkingLevelChangeEntry(SessionEntryBase):
    """Entry recording a thinking level change."""
    thinking_level: str

    def __init__(
        self,
        id: str,
        parent_id: Optional[str],
        timestamp: str,
        thinking_level: str,
    ):
        super().__init__(
            type="thinking_level_change",
            id=id,
            parent_id=parent_id,
            timestamp=timestamp,
        )
        self.thinking_level = thinking_level


@dataclass
class ModelChangeEntry(SessionEntryBase):
    """Entry recording a model change."""
    provider: str
    model_id: str

    def __init__(
        self,
        id: str,
        parent_id: Optional[str],
        timestamp: str,
        provider: str,
        model_id: str,
    ):
        super().__init__(
            type="model_change",
            id=id,
            parent_id=parent_id,
            timestamp=timestamp,
        )
        self.provider = provider
        self.model_id = model_id


@dataclass
class CompactionEntry(SessionEntryBase):
    """Entry recording context compaction.

    When context grows too large, old messages can be compacted into a summary.
    The first_kept_entry_id marks where the kept history starts.
    """
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    details: Optional[Dict[str, Any]] = None
    from_hook: bool = False

    def __init__(
        self,
        id: str,
        parent_id: Optional[str],
        timestamp: str,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: Optional[Dict[str, Any]] = None,
        from_hook: bool = False,
    ):
        super().__init__(
            type="compaction",
            id=id,
            parent_id=parent_id,
            timestamp=timestamp,
        )
        self.summary = summary
        self.first_kept_entry_id = first_kept_entry_id
        self.tokens_before = tokens_before
        self.details = details
        self.from_hook = from_hook


@dataclass
class BranchSummaryEntry(SessionEntryBase):
    """Entry summarizing an abandoned conversation branch.

    When branching from an earlier point, this captures context
    from the path that was abandoned.
    """
    from_id: str
    summary: str
    details: Optional[Dict[str, Any]] = None
    from_hook: bool = False

    def __init__(
        self,
        id: str,
        parent_id: Optional[str],
        timestamp: str,
        from_id: str,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        from_hook: bool = False,
    ):
        super().__init__(
            type="branch_summary",
            id=id,
            parent_id=parent_id,
            timestamp=timestamp,
        )
        self.from_id = from_id
        self.summary = summary
        self.details = details
        self.from_hook = from_hook


@dataclass
class CustomEntry(SessionEntryBase):
    """Entry for extensions to store custom data.

    Extensions can use custom_type to identify their entries.
    Does NOT participate in LLM context.
    """
    custom_type: str
    data: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        id: str,
        parent_id: Optional[str],
        timestamp: str,
        custom_type: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            type="custom",
            id=id,
            parent_id=parent_id,
            timestamp=timestamp,
        )
        self.custom_type = custom_type
        self.data = data


@dataclass
class LabelEntry(SessionEntryBase):
    """Entry for user-defined bookmarks/markers on entries."""
    target_id: str
    label: Optional[str] = None

    def __init__(
        self,
        id: str,
        parent_id: Optional[str],
        timestamp: str,
        target_id: str,
        label: Optional[str] = None,
    ):
        super().__init__(
            type="label",
            id=id,
            parent_id=parent_id,
            timestamp=timestamp,
        )
        self.target_id = target_id
        self.label = label


@dataclass
class SessionInfoEntry(SessionEntryBase):
    """Entry for session metadata (e.g., user-defined display name)."""
    name: Optional[str] = None

    def __init__(
        self,
        id: str,
        parent_id: Optional[str],
        timestamp: str,
        name: Optional[str] = None,
    ):
        super().__init__(
            type="session_info",
            id=id,
            parent_id=parent_id,
            timestamp=timestamp,
        )
        self.name = name


# Union type for all session entries
SessionEntry = Union[
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
    ModelChangeEntry,
    CompactionEntry,
    BranchSummaryEntry,
    CustomEntry,
    LabelEntry,
    SessionInfoEntry,
]

# Union type for file entries (header + session entries)
FileEntry = Union[SessionHeader, SessionEntry]


def parse_session_entry(d: Dict[str, Any]) -> SessionEntry:
    """Parse a session entry from a dictionary.

    Args:
        d: Dictionary parsed from JSON line.

    Returns:
        Appropriate SessionEntry subclass based on type field.

    Raises:
        ValueError: If entry type is unknown or missing.
    """
    entry_type = d.get("type")
    if entry_type == "message":
        return SessionMessageEntry(
            id=d["id"],
            parent_id=d.get("parent_id"),
            timestamp=d["timestamp"],
            message=d["message"],
        )
    elif entry_type == "thinking_level_change":
        return ThinkingLevelChangeEntry(
            id=d["id"],
            parent_id=d.get("parent_id"),
            timestamp=d["timestamp"],
            thinking_level=d["thinking_level"],
        )
    elif entry_type == "model_change":
        return ModelChangeEntry(
            id=d["id"],
            parent_id=d.get("parent_id"),
            timestamp=d["timestamp"],
            provider=d["provider"],
            model_id=d["model_id"],
        )
    elif entry_type == "compaction":
        return CompactionEntry(
            id=d["id"],
            parent_id=d.get("parent_id"),
            timestamp=d["timestamp"],
            summary=d["summary"],
            first_kept_entry_id=d["first_kept_entry_id"],
            tokens_before=d["tokens_before"],
            details=d.get("details"),
            from_hook=d.get("from_hook", False),
        )
    elif entry_type == "branch_summary":
        return BranchSummaryEntry(
            id=d["id"],
            parent_id=d.get("parent_id"),
            timestamp=d["timestamp"],
            from_id=d["from_id"],
            summary=d["summary"],
            details=d.get("details"),
            from_hook=d.get("from_hook", False),
        )
    elif entry_type == "custom":
        return CustomEntry(
            id=d["id"],
            parent_id=d.get("parent_id"),
            timestamp=d["timestamp"],
            custom_type=d["custom_type"],
            data=d.get("data"),
        )
    elif entry_type == "label":
        return LabelEntry(
            id=d["id"],
            parent_id=d.get("parent_id"),
            timestamp=d["timestamp"],
            target_id=d["target_id"],
            label=d.get("label"),
        )
    elif entry_type == "session_info":
        return SessionInfoEntry(
            id=d["id"],
            parent_id=d.get("parent_id"),
            timestamp=d["timestamp"],
            name=d.get("name"),
        )
    else:
        raise ValueError(f"Unknown entry type: {entry_type}")


def parse_file_entry(d: Dict[str, Any]) -> FileEntry:
    """Parse a file entry (header or session entry) from a dictionary.

    Args:
        d: Dictionary parsed from JSON line.

    Returns:
        SessionHeader or SessionEntry subclass.

    Raises:
        ValueError: If entry type is unknown or missing.
    """
    entry_type = d.get("type")
    if entry_type == "session":
        return SessionHeader.from_dict(d)
    else:
        return parse_session_entry(d)


def serialize_entry(entry: FileEntry) -> str:
    """Serialize an entry to a JSON line.

    Args:
        entry: SessionHeader or SessionEntry.

    Returns:
        JSON string for writing to session file.
    """
    if isinstance(entry, SessionHeader):
        return json.dumps(entry.to_dict())
    elif hasattr(entry, "to_dict"):
        return json.dumps(entry.to_dict())
    else:
        return json.dumps(asdict(entry))