"""
Session Manager for pi-coding-agent.

Manages conversation sessions as append-only trees stored in JSONL files.
Each session entry has an id and parentId forming a tree structure.
The "leaf" pointer tracks the current position in the tree.

Key features:
- JSONL format for efficient append-only storage
- Tree structure with parentId references for branching
- Leaf pointer for current position
- Migration support (v1->v2->v3)
- Session persistence and loading
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple

from .session_header import (
    SessionHeader,
    SessionEntry,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
    ModelChangeEntry,
    CompactionEntry,
    BranchSummaryEntry,
    CustomEntry,
    LabelEntry,
    SessionInfoEntry,
    FileEntry,
    SessionVersion,
    CURRENT_SESSION_VERSION,
    parse_file_entry,
    serialize_entry,
)


def create_session_id() -> str:
    """Generate a unique session ID using UUID7-like format."""
    # Use timestamp-based ID for natural ordering
    timestamp = int(datetime.now().timestamp() * 1000)
    random_part = uuid.uuid4().hex[:8]
    return f"{timestamp:x}-{random_part}"


def generate_id(existing_ids: Set[str]) -> str:
    """Generate a unique short ID (8 hex chars, collision-checked)."""
    for _ in range(100):
        id = uuid.uuid4().hex[:8]
        if id not in existing_ids:
            return id
    # Fallback to full UUID if somehow we have collisions
    return uuid.uuid4().hex


def parse_session_entries(content: str) -> List[FileEntry]:
    """Parse session entries from JSONL content.

    Args:
        content: Raw JSONL file content.

    Returns:
        List of parsed FileEntry objects (header + entries).
    """
    entries: List[FileEntry] = []
    lines = content.strip().split("\n")

    for line in lines:
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            entry = parse_file_entry(d)
            entries.append(entry)
        except (json.JSONDecodeError, ValueError):
            # Skip malformed lines
            continue

    return entries


def load_entries_from_file(file_path: str) -> List[FileEntry]:
    """Load entries from a session file.

    Args:
        file_path: Path to the session file.

    Returns:
        List of parsed FileEntry objects, empty list if file doesn't exist.
    """
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except IOError:
        return []

    entries = parse_session_entries(content)

    # Validate session header
    if len(entries) == 0:
        return entries

    header = entries[0]
    if not isinstance(header, SessionHeader):
        return []

    return entries


def migrate_v1_to_v2(entries: List[FileEntry]) -> None:
    """Migrate v1 entries to v2: add id/parentId tree structure.

    Mutates entries in place.
    """
    existing_ids: Set[str] = set()
    prev_id: Optional[str] = None

    for entry in entries:
        if isinstance(entry, SessionHeader):
            entry.version = SessionVersion.V2
            continue

        # Add id and parentId
        if hasattr(entry, "id") and not entry.id:
            entry.id = generate_id(existing_ids)
        existing_ids.add(entry.id)

        if hasattr(entry, "parent_id"):
            entry.parent_id = prev_id

        prev_id = entry.id

        # Convert first_kept_entry_index to first_kept_entry_id for compaction
        if isinstance(entry, CompactionEntry):
            # Handle legacy format if present
            if hasattr(entry, "first_kept_entry_index"):
                idx = entry.first_kept_entry_index
                if isinstance(idx, int) and idx < len(entries):
                    target = entries[idx]
                    if not isinstance(target, SessionHeader):
                        entry.first_kept_entry_id = target.id
                # Remove legacy field
                if hasattr(entry, "__dict__"):
                    entry.__dict__.pop("first_kept_entry_index", None)


def migrate_v2_to_v3(entries: List[FileEntry]) -> None:
    """Migrate v2 entries to v3: rename hookMessage role to custom.

    Mutates entries in place.
    """
    for entry in entries:
        if isinstance(entry, SessionHeader):
            entry.version = SessionVersion.V3
            continue

        # Update message entries with hookMessage role
        if isinstance(entry, SessionMessageEntry):
            if entry.message.get("role") == "hookMessage":
                entry.message["role"] = "custom"


def migrate_to_current_version(entries: List[FileEntry]) -> bool:
    """Run all necessary migrations to bring entries to current version.

    Args:
        entries: List of FileEntry objects (mutated in place).

    Returns:
        True if any migration was applied.
    """
    if len(entries) == 0:
        return False

    header = entries[0]
    if not isinstance(header, SessionHeader):
        return False

    version = header.version

    if version.value >= CURRENT_SESSION_VERSION.value:
        return False

    if version.value < 2:
        migrate_v1_to_v2(entries)
    if version.value < 3:
        migrate_v2_to_v3(entries)

    return True


def get_latest_compaction_entry(entries: List[SessionEntry]) -> Optional[CompactionEntry]:
    """Find the latest compaction entry in the list.

    Args:
        entries: List of SessionEntry objects.

    Returns:
        Latest CompactionEntry or None if not found.
    """
    for i in range(len(entries) - 1, -1, -1):
        if isinstance(entries[i], CompactionEntry):
            return entries[i]
    return None


class SessionTreeNode:
    """Tree node for get_tree() - defensive copy of session structure."""

    def __init__(
        self,
        entry: SessionEntry,
        children: List["SessionTreeNode"] = None,
        label: Optional[str] = None,
        label_timestamp: Optional[str] = None,
    ):
        self.entry = entry
        self.children: List["SessionTreeNode"] = children or []
        self.label: Optional[str] = label
        self.label_timestamp: Optional[str] = label_timestamp


class SessionContext:
    """Session context built from entries for LLM."""

    def __init__(
        self,
        messages: List[Dict[str, Any]],
        thinking_level: str = "off",
        model: Optional[Dict[str, str]] = None,
    ):
        self.messages = messages
        self.thinking_level = thinking_level
        self.model = model  # {"provider": str, "model_id": str}


class SessionInfo:
    """Information about a session file."""

    def __init__(
        self,
        path: str,
        id: str,
        cwd: str,
        created: datetime,
        modified: datetime,
        message_count: int,
        first_message: str,
        all_messages_text: str,
        name: Optional[str] = None,
        parent_session_path: Optional[str] = None,
    ):
        self.path = path
        self.id = id
        self.cwd = cwd
        self.name = name
        self.parent_session_path = parent_session_path
        self.created = created
        self.modified = modified
        self.message_count = message_count
        self.first_message = first_message
        self.all_messages_text = all_messages_text


def build_session_context(
    entries: List[SessionEntry],
    leaf_id: Optional[str] = None,
    by_id: Optional[Dict[str, SessionEntry]] = None,
) -> SessionContext:
    """Build the session context from entries using tree traversal.

    If leaf_id is provided, walks from that entry to root.
    Handles compaction and branch summaries along the path.

    Args:
        entries: List of SessionEntry objects.
        leaf_id: Optional leaf entry ID to start traversal from.
        by_id: Optional pre-built id->entry mapping.

    Returns:
        SessionContext with messages, thinking_level, and model info.
    """
    # Build uuid index if not available
    if by_id is None:
        by_id = {}
        for entry in entries:
            by_id[entry.id] = entry

    # Find leaf
    leaf: Optional[SessionEntry] = None
    if leaf_id is None:
        # Fallback to last entry
        if entries:
            leaf = entries[-1]
    elif leaf_id is not None:
        leaf = by_id.get(leaf_id)

    if leaf is None:
        return SessionContext(messages=[], thinking_level="off", model=None)

    # Walk from leaf to root, collecting path
    path: List[SessionEntry] = []
    current: Optional[SessionEntry] = leaf
    while current is not None:
        path.insert(0, current)
        parent_id = current.parent_id
        current = by_id.get(parent_id) if parent_id else None

    # Extract settings and find compaction
    thinking_level = "off"
    model: Optional[Dict[str, str]] = None
    compaction: Optional[CompactionEntry] = None

    for entry in path:
        if isinstance(entry, ThinkingLevelChangeEntry):
            thinking_level = entry.thinking_level
        elif isinstance(entry, ModelChangeEntry):
            model = {"provider": entry.provider, "model_id": entry.model_id}
        elif isinstance(entry, SessionMessageEntry):
            msg = entry.message
            if msg.get("role") == "assistant":
                model = {
                    "provider": msg.get("provider", ""),
                    "model_id": msg.get("model", ""),
                }
        elif isinstance(entry, CompactionEntry):
            compaction = entry

    # Build messages
    messages: List[Dict[str, Any]] = []

    def append_message(entry: SessionEntry):
        """Append message from entry to messages list."""
        if isinstance(entry, SessionMessageEntry):
            messages.append(entry.message)
        elif isinstance(entry, BranchSummaryEntry) and entry.summary:
            # Create branch summary message
            messages.append({
                "role": "user",
                "content": f"[Branch summary: {entry.summary}]",
                "timestamp": entry.timestamp,
            })
        # Custom message entries would be handled here if needed

    if compaction:
        # Emit summary first
        messages.append({
            "role": "user",
            "content": f"[Context compaction summary: {compaction.summary}]",
            "timestamp": compaction.timestamp,
        })

        # Find compaction index in path
        compaction_idx = -1
        for i, e in enumerate(path):
            if isinstance(e, CompactionEntry) and e.id == compaction.id:
                compaction_idx = i
                break

        # Emit kept messages (before compaction, starting from first_kept_entry_id)
        found_first_kept = False
        for i in range(compaction_idx):
            entry = path[i]
            if entry.id == compaction.first_kept_entry_id:
                found_first_kept = True
            if found_first_kept:
                append_message(entry)

        # Emit messages after compaction
        for i in range(compaction_idx + 1, len(path)):
            entry = path[i]
            append_message(entry)
    else:
        # No compaction - emit all messages
        for entry in path:
            append_message(entry)

    return SessionContext(messages=messages, thinking_level=thinking_level, model=model)


class SessionManager:
    """Manages conversation sessions as append-only trees stored in JSONL files.

    Each session entry has an id and parentId forming a tree structure.
    The "leaf" pointer tracks the current position. Appending creates a child
    of the current leaf. Branching moves the leaf to an earlier entry,
    allowing new branches without modifying history.

    Use build_session_context() to get the resolved message list for the LLM,
    which handles compaction summaries and follows the path from root to
    current leaf.
    """

    def __init__(
        self,
        cwd: str,
        session_dir: str,
        session_file: Optional[str] = None,
        persist: bool = True,
    ):
        """Initialize SessionManager.

        Args:
            cwd: Working directory (stored in session header).
            session_dir: Directory for session files.
            session_file: Optional specific session file to open.
            persist: Whether to persist to disk (False for in-memory mode).
        """
        self._cwd = cwd
        self._session_dir = session_dir
        self._persist = persist
        self._flushed = False
        self._session_id: str = ""
        self._session_file: Optional[str] = None
        self._file_entries: List[FileEntry] = []
        self._by_id: Dict[str, SessionEntry] = {}
        self._labels_by_id: Dict[str, str] = {}
        self._label_timestamps_by_id: Dict[str, str] = {}
        self._leaf_id: Optional[str] = None

        if persist and session_dir and not os.path.exists(session_dir):
            os.makedirs(session_dir, exist_ok=True)

        if session_file:
            self._set_session_file(session_file)
        else:
            self._new_session()

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _set_session_file(self, session_file: str) -> None:
        """Switch to a different session file (used for resume and branching)."""
        self._session_file = os.path.abspath(session_file)

        if os.path.exists(self._session_file):
            self._file_entries = load_entries_from_file(self._session_file)

            # If file was empty or corrupted, start fresh
            if len(self._file_entries) == 0:
                explicit_path = self._session_file
                self._new_session()
                self._session_file = explicit_path
                self._rewrite_file()
                self._flushed = True
                return

            header = self._file_entries[0]
            if isinstance(header, SessionHeader):
                self._session_id = header.session_id
            else:
                self._session_id = create_session_id()

            if migrate_to_current_version(self._file_entries):
                self._rewrite_file()

            self._build_index()
            self._flushed = True
        else:
            explicit_path = self._session_file
            self._new_session()
            self._session_file = explicit_path

    def _new_session(self, options: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a new session with fresh header."""
        self._session_id = options.get("id") if options else None
        if not self._session_id:
            self._session_id = create_session_id()

        timestamp = datetime.now().isoformat()
        header = SessionHeader(
            version=CURRENT_SESSION_VERSION,
            session_id=self._session_id,
            model_id="",
            provider="",
            created_at=int(datetime.now().timestamp() * 1000),
            updated_at=int(datetime.now().timestamp() * 1000),
            cwd=self._cwd,
            parent_session=options.get("parent_session") if options else None,
        )

        self._file_entries = [header]
        self._by_id.clear()
        self._labels_by_id.clear()
        self._label_timestamps_by_id.clear()
        self._leaf_id = None
        self._flushed = False

        if self._persist:
            file_timestamp = timestamp.replace(":", "-").replace(".", "-")
            self._session_file = os.path.join(
                self._get_session_dir(),
                f"{file_timestamp}_{self._session_id}.jsonl",
            )

        return self._session_file

    def _build_index(self) -> None:
        """Build internal index from file entries."""
        self._by_id.clear()
        self._labels_by_id.clear()
        self._label_timestamps_by_id.clear()
        self._leaf_id = None

        # Get header's leaf_entry_id if available
        header_leaf = self._header.leaf_entry_id if self._header else None

        for entry in self._file_entries:
            if isinstance(entry, SessionHeader):
                continue

            self._by_id[entry.id] = entry

            # Track last entry as fallback for leaf
            self._leaf_id = entry.id

            if isinstance(entry, LabelEntry):
                if entry.label:
                    self._labels_by_id[entry.target_id] = entry.label
                    self._label_timestamps_by_id[entry.target_id] = entry.timestamp
                else:
                    self._labels_by_id.pop(entry.target_id, None)
                    self._label_timestamps_by_id.pop(entry.target_id, None)

        # Use header's leaf if valid, otherwise use last entry (fallback)
        if header_leaf and header_leaf in self._by_id:
            self._leaf_id = header_leaf

    @property
    def _header(self) -> Optional[SessionHeader]:
        """Get the session header from file entries."""
        for entry in self._file_entries:
            if isinstance(entry, SessionHeader):
                return entry
        return None

    def _update_header(self) -> None:
        """Update header fields and persist if needed.

        Updates:
        - updated_at timestamp
        - leaf_entry_id to current leaf position
        """
        header = self._header
        if not header:
            return

        # Update timestamp
        header.updated_at = int(datetime.now().timestamp() * 1000)

        # Update leaf position
        header.leaf_entry_id = self._leaf_id

        # Persist header changes by rewriting file
        if self._persist and self._session_file and self._flushed:
            self._rewrite_file_with_header()

    def _rewrite_file_with_header(self) -> None:
        """Rewrite file with updated header while preserving entries."""
        if not self._persist or not self._session_file:
            return

        # Rewrite entire file with updated header
        self._rewrite_file()

    def _rewrite_file(self) -> None:
        """Rewrite entire session file."""
        if not self._persist or not self._session_file:
            return

        lines = [serialize_entry(e) for e in self._file_entries]
        content = "\n".join(lines) + "\n"

        with open(self._session_file, "w", encoding="utf-8") as f:
            f.write(content)

    def _persist_entry(self, entry: SessionEntry) -> None:
        """Persist entry to file."""
        if not self._persist or not self._session_file:
            return

        # Check if there's an assistant message
        has_assistant = any(
            isinstance(e, SessionMessageEntry) and e.message.get("role") == "assistant"
            for e in self._file_entries
        )

        if not has_assistant:
            # Mark as not flushed so when assistant arrives, all entries get written
            self._flushed = False
            return

        if not self._flushed:
            # Write all entries
            with open(self._session_file, "a", encoding="utf-8") as f:
                for e in self._file_entries:
                    f.write(serialize_entry(e) + "\n")
            self._flushed = True
        else:
            # Append single entry
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write(serialize_entry(entry) + "\n")

    def _append_entry(self, entry: SessionEntry) -> None:
        """Append entry to session and persist."""
        self._file_entries.append(entry)
        self._by_id[entry.id] = entry
        self._leaf_id = entry.id
        self._persist_entry(entry)
        self._update_header()

    def _generate_entry_id(self) -> str:
        """Generate a unique entry ID."""
        existing = set(self._by_id.keys())
        return generate_id(existing)

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------

    def is_persisted(self) -> bool:
        """Check if session is persisted to disk."""
        return self._persist

    def get_cwd(self) -> str:
        """Get working directory."""
        return self._cwd

    def _get_session_dir(self) -> str:
        """Get session directory."""
        return self._session_dir

    def get_session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    def get_session_file(self) -> Optional[str]:
        """Get session file path."""
        return self._session_file

    # -------------------------------------------------------------------------
    # Entry Appending Methods
    # -------------------------------------------------------------------------

    def append_message(self, message: Dict[str, Any]) -> str:
        """Append a message as child of current leaf.

        Args:
            message: Message dictionary with role, content, etc.

        Returns:
            Entry ID of the appended message.
        """
        entry = SessionMessageEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=datetime.now().isoformat(),
            message=message,
        )
        self._append_entry(entry)
        return entry.id

    def append_thinking_level_change(self, thinking_level: str) -> str:
        """Append a thinking level change entry.

        Args:
            thinking_level: New thinking level value.

        Returns:
            Entry ID.
        """
        entry = ThinkingLevelChangeEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=datetime.now().isoformat(),
            thinking_level=thinking_level,
        )
        self._append_entry(entry)
        return entry.id

    def append_model_change(self, provider: str, model_id: str) -> str:
        """Append a model change entry.

        Args:
            provider: Provider name.
            model_id: Model identifier.

        Returns:
            Entry ID.
        """
        entry = ModelChangeEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model_id=model_id,
        )
        self._append_entry(entry)
        return entry.id

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: Optional[Dict[str, Any]] = None,
        from_hook: bool = False,
    ) -> str:
        """Append a compaction entry.

        Args:
            summary: Summary text for compacted messages.
            first_kept_entry_id: ID of first entry kept after compaction.
            tokens_before: Token count before compaction.
            details: Optional extension-specific data.
            from_hook: True if generated by extension hook.

        Returns:
            Entry ID.
        """
        entry = CompactionEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=datetime.now().isoformat(),
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
            from_hook=from_hook,
        )
        # Add compaction entry ID to header's compaction_entries list
        header = self._header
        if header:
            header.compaction_entries.append(entry.id)
        self._append_entry(entry)
        return entry.id

    def append_custom_entry(self, custom_type: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Append a custom entry for extensions.

        Args:
            custom_type: Extension identifier.
            data: Optional extension data.

        Returns:
            Entry ID.
        """
        entry = CustomEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=datetime.now().isoformat(),
            custom_type=custom_type,
            data=data,
        )
        self._append_entry(entry)
        return entry.id

    def append_session_info(self, name: str) -> str:
        """Append a session info entry with display name.

        Args:
            name: Display name for the session.

        Returns:
            Entry ID.
        """
        entry = SessionInfoEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=datetime.now().isoformat(),
            name=name.strip(),
        )
        self._append_entry(entry)
        return entry.id

    def append_label_change(self, target_id: str, label: Optional[str]) -> str:
        """Set or clear a label on an entry.

        Args:
            target_id: Entry ID to label.
            label: Label text, or None/empty to clear.

        Returns:
            Entry ID of the label entry.

        Raises:
            ValueError: If target entry doesn't exist.
        """
        if target_id not in self._by_id:
            raise ValueError(f"Entry {target_id} not found")

        entry = LabelEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=datetime.now().isoformat(),
            target_id=target_id,
            label=label,
        )
        self._append_entry(entry)

        if label:
            self._labels_by_id[target_id] = label
            self._label_timestamps_by_id[target_id] = entry.timestamp
        else:
            self._labels_by_id.pop(target_id, None)
            self._label_timestamps_by_id.pop(target_id, None)

        return entry.id

    def append_branch_summary(
        self,
        from_id: Optional[str],
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        from_hook: bool = False,
    ) -> str:
        """Append a branch summary entry.

        Args:
            from_id: Entry ID where branch started (None for root).
            summary: Summary of abandoned path.
            details: Optional extension data.
            from_hook: True if generated by extension hook.

        Returns:
            Entry ID.

        Raises:
            ValueError: If from_id entry doesn't exist.
        """
        if from_id is not None and from_id not in self._by_id:
            raise ValueError(f"Entry {from_id} not found")

        self._leaf_id = from_id

        entry = BranchSummaryEntry(
            id=self._generate_entry_id(),
            parent_id=from_id,
            timestamp=datetime.now().isoformat(),
            from_id=from_id or "root",
            summary=summary,
            details=details,
            from_hook=from_hook,
        )
        self._append_entry(entry)
        return entry.id

    # -------------------------------------------------------------------------
    # Tree Traversal Methods
    # -------------------------------------------------------------------------

    def get_leaf_id(self) -> Optional[str]:
        """Get current leaf entry ID."""
        return self._leaf_id

    def get_leaf_entry(self) -> Optional[SessionEntry]:
        """Get current leaf entry."""
        if self._leaf_id:
            return self._by_id.get(self._leaf_id)
        return None

    def get_entry(self, id: str) -> Optional[SessionEntry]:
        """Get entry by ID."""
        return self._by_id.get(id)

    def get_children(self, parent_id: str) -> List[SessionEntry]:
        """Get all direct children of an entry.

        Args:
            parent_id: Parent entry ID.

        Returns:
            List of child SessionEntry objects.
        """
        children: List[SessionEntry] = []
        for entry in self._by_id.values():
            if entry.parent_id == parent_id:
                children.append(entry)
        return children

    def get_label(self, id: str) -> Optional[str]:
        """Get label for an entry, if any.

        Args:
            id: Entry ID.

        Returns:
            Label string or None.
        """
        return self._labels_by_id.get(id)

    def get_branch(self, from_id: Optional[str] = None) -> List[SessionEntry]:
        """Walk from entry to root, returning all entries in path order.

        Args:
            from_id: Starting entry ID (defaults to current leaf).

        Returns:
            List of entries from root to from_id.
        """
        path: List[SessionEntry] = []
        start_id = from_id or self._leaf_id

        current = self._by_id.get(start_id) if start_id else None
        while current is not None:
            path.insert(0, current)
            parent_id = current.parent_id
            current = self._by_id.get(parent_id) if parent_id else None

        return path

    def build_session_context(self) -> SessionContext:
        """Build the session context (what gets sent to the LLM).

        Uses tree traversal from current leaf.
        """
        entries = self.get_entries()
        return build_session_context(entries, self._leaf_id, self._by_id)

    def get_header(self) -> Optional[SessionHeader]:
        """Get session header."""
        for entry in self._file_entries:
            if isinstance(entry, SessionHeader):
                return entry
        return None

    def get_entries(self) -> List[SessionEntry]:
        """Get all session entries (excludes header).

        Returns:
            Shallow copy of session entries.
        """
        return [e for e in self._file_entries if not isinstance(e, SessionHeader)]

    def get_tree(self) -> List[SessionTreeNode]:
        """Get session as tree structure.

        Returns:
            List of root SessionTreeNode objects with children populated.
        """
        entries = self.get_entries()
        node_map: Dict[str, SessionTreeNode] = {}
        roots: List[SessionTreeNode] = []

        # Create nodes with resolved labels
        for entry in entries:
            label = self._labels_by_id.get(entry.id)
            label_timestamp = self._label_timestamps_by_id.get(entry.id)
            node_map[entry.id] = SessionTreeNode(entry, [], label, label_timestamp)

        # Build tree
        for entry in entries:
            node = node_map.get(entry.id)
            if node is None:
                continue

            if entry.parent_id is None or entry.parent_id == entry.id:
                roots.append(node)
            else:
                parent = node_map.get(entry.parent_id)
                if parent:
                    parent.children.append(node)
                else:
                    # Orphan - treat as root
                    roots.append(node)

        # Sort children by timestamp (oldest first)
        stack = list(roots)
        while stack:
            node = stack.pop()
            node.children.sort(
                key=lambda n: datetime.fromisoformat(n.entry.timestamp).timestamp()
            )
            stack.extend(node.children)

        return roots

    def get_session_name(self) -> Optional[str]:
        """Get session name from latest session_info entry."""
        entries = self.get_entries()
        for i in range(len(entries) - 1, -1, -1):
            entry = entries[i]
            if isinstance(entry, SessionInfoEntry):
                name = entry.name
                return name.strip() if name else None
        return None

    # -------------------------------------------------------------------------
    # Branching Methods
    # -------------------------------------------------------------------------

    def branch(self, branch_from_id: str) -> None:
        """Start a new branch from an earlier entry.

        Moves the leaf pointer to the specified entry. The next append
        will create a child of that entry, forming a new branch.

        Args:
            branch_from_id: Entry ID to branch from.

        Raises:
            ValueError: If entry doesn't exist.
        """
        if branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id
        self._update_header()

    def reset_leaf(self) -> None:
        """Reset leaf pointer to null (before any entries).

        The next append will create a new root entry (parentId = null).
        """
        self._leaf_id = None
        self._update_header()

    def create_branched_session(self, leaf_id: str) -> Optional[str]:
        """Create a new session file with path from root to leaf.

        Args:
            leaf_id: Leaf entry ID to use as end of path.

        Returns:
            New session file path, or None if not persisting.

        Raises:
            ValueError: If leaf entry doesn't exist.
        """
        previous_session_file = self._session_file
        path = self.get_branch(leaf_id)

        if len(path) == 0:
            raise ValueError(f"Entry {leaf_id} not found")

        # Filter out label entries
        path_without_labels = [e for e in path if not isinstance(e, LabelEntry)]

        new_session_id = create_session_id()
        timestamp = datetime.now().isoformat()
        file_timestamp = timestamp.replace(":", "-").replace(".", "-")
        new_session_file = os.path.join(
            self._get_session_dir(),
            f"{file_timestamp}_{new_session_id}.jsonl",
        )

        header = SessionHeader(
            version=CURRENT_SESSION_VERSION,
            session_id=new_session_id,
            model_id="",
            provider="",
            created_at=int(datetime.now().timestamp() * 1000),
            updated_at=int(datetime.now().timestamp() * 1000),
            cwd=self._cwd,
            parent_session=previous_session_file if self._persist else None,
        )

        # Collect labels for entries in path
        path_entry_ids = set(e.id for e in path_without_labels)
        labels_to_write: List[Tuple[str, str, str]] = []
        for target_id, label in self._labels_by_id.items():
            if target_id in path_entry_ids:
                label_timestamp = self._label_timestamps_by_id.get(target_id, "")
                labels_to_write.append((target_id, label, label_timestamp))

        if self._persist:
            # Build label entries
            last_entry_id = path_without_labels[-1].id if path_without_labels else None
            parent_id = last_entry_id
            label_entries: List[LabelEntry] = []
            all_ids = set(path_entry_ids)

            for target_id, label, label_timestamp in labels_to_write:
                label_entry_id = generate_id(all_ids)
                all_ids.add(label_entry_id)
                label_entry = LabelEntry(
                    id=label_entry_id,
                    parent_id=parent_id,
                    timestamp=label_timestamp,
                    target_id=target_id,
                    label=label,
                )
                label_entries.append(label_entry)
                parent_id = label_entry.id

            self._file_entries = [header] + path_without_labels + label_entries
            self._session_id = new_session_id
            self._session_file = new_session_file
            self._build_index()

            # Write file if has assistant message
            has_assistant = any(
                isinstance(e, SessionMessageEntry) and e.message.get("role") == "assistant"
                for e in self._file_entries
            )
            if has_assistant:
                self._rewrite_file()
                self._flushed = True
            else:
                self._flushed = False

            return new_session_file

        # In-memory mode
        last_entry_id = path_without_labels[-1].id if path_without_labels else None
        parent_id = last_entry_id
        label_entries: List[LabelEntry] = []
        all_ids = set(path_entry_ids)

        for target_id, label, label_timestamp in labels_to_write:
            label_entry_id = generate_id(all_ids)
            all_ids.add(label_entry_id)
            label_entry = LabelEntry(
                id=label_entry_id,
                parent_id=parent_id,
                timestamp=label_timestamp,
                target_id=target_id,
                label=label,
            )
            label_entries.append(label_entry)
            parent_id = label_entry.id

        self._file_entries = [header] + path_without_labels + label_entries
        self._session_id = new_session_id
        self._build_index()

        return None

    # -------------------------------------------------------------------------
    # Static Factory Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def create(cwd: str, session_dir: Optional[str] = None) -> "SessionManager":
        """Create a new session.

        Args:
            cwd: Working directory.
            session_dir: Optional session directory.

        Returns:
            New SessionManager instance.
        """
        dir_path = session_dir or SessionManager._get_default_session_dir(cwd)
        return SessionManager(cwd, dir_path, None, True)

    @staticmethod
    def open(
        path: str,
        session_dir: Optional[str] = None,
        cwd_override: Optional[str] = None,
    ) -> "SessionManager":
        """Open an existing session file.

        Args:
            path: Path to session file.
            session_dir: Optional session directory.
            cwd_override: Optional cwd override.

        Returns:
            SessionManager instance with loaded session.
        """
        entries = load_entries_from_file(path)
        header = None
        for e in entries:
            if isinstance(e, SessionHeader):
                header = e
                break

        cwd = cwd_override or (header.cwd if header else os.getcwd())
        dir_path = session_dir or os.path.dirname(os.path.abspath(path))

        return SessionManager(cwd, dir_path, path, True)

    @staticmethod
    def continue_recent(cwd: str, session_dir: Optional[str] = None) -> "SessionManager":
        """Continue the most recent session, or create new.

        Args:
            cwd: Working directory.
            session_dir: Optional session directory.

        Returns:
            SessionManager instance.
        """
        dir_path = session_dir or SessionManager._get_default_session_dir(cwd)
        most_recent = SessionManager._find_most_recent_session(dir_path)

        if most_recent:
            return SessionManager(cwd, dir_path, most_recent, True)
        return SessionManager(cwd, dir_path, None, True)

    @staticmethod
    def in_memory(cwd: str = None) -> "SessionManager":
        """Create an in-memory session (no file persistence).

        Args:
            cwd: Working directory (defaults to current directory).

        Returns:
            SessionManager instance without persistence.
        """
        return SessionManager(cwd or os.getcwd(), "", None, False)

    @staticmethod
    def _get_default_session_dir(cwd: str) -> str:
        """Compute default session directory for a cwd.

        Args:
            cwd: Working directory.

        Returns:
            Path to session directory.
        """
        # Encode cwd into safe directory name
        safe_path = cwd.replace("/", "-").replace("\\", "-").replace(":", "-")
        safe_path = f"--{safe_path.lstrip('-')}--"

        # Use home directory for sessions
        home = os.path.expanduser("~")
        agent_dir = os.path.join(home, ".pi", "agent", "sessions")
        session_dir = os.path.join(agent_dir, safe_path)

        if not os.path.exists(session_dir):
            os.makedirs(session_dir, exist_ok=True)

        return session_dir

    @staticmethod
    def _find_most_recent_session(session_dir: str) -> Optional[str]:
        """Find the most recent valid session file in directory.

        Args:
            session_dir: Directory to search.

        Returns:
            Path to most recent session file, or None.
        """
        if not os.path.exists(session_dir):
            return None

        try:
            files = [
                f for f in os.listdir(session_dir)
                if f.endswith(".jsonl")
            ]
            file_paths = [os.path.join(session_dir, f) for f in files]

            # Filter valid session files and sort by mtime
            valid_files = []
            for fp in file_paths:
                if SessionManager._is_valid_session_file(fp):
                    mtime = os.path.getmtime(fp)
                    valid_files.append((fp, mtime))

            valid_files.sort(key=lambda x: x[1], reverse=True)
            return valid_files[0][0] if valid_files else None
        except OSError:
            return None

    @staticmethod
    def _is_valid_session_file(file_path: str) -> bool:
        """Check if file is a valid session file.

        Args:
            file_path: Path to check.

        Returns:
            True if file has valid session header.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if not first_line:
                return False
            header = json.loads(first_line)
            return header.get("type") == "session" and "id" in header
        except (IOError, json.JSONDecodeError):
            return False

    @staticmethod
    async def list_sessions(
        cwd: str,
        session_dir: Optional[str] = None,
    ) -> List[SessionInfo]:
        """List all sessions for a directory.

        Args:
            cwd: Working directory.
            session_dir: Optional session directory override.

        Returns:
            List of SessionInfo objects, sorted by modified time (newest first).
        """
        dir_path = session_dir or SessionManager._get_default_session_dir(cwd)
        return SessionManager._list_sessions_from_dir(dir_path)

    @staticmethod
    def _list_sessions_from_dir(dir_path: str) -> List[SessionInfo]:
        """List sessions from a specific directory.

        Args:
            dir_path: Directory path.

        Returns:
            List of SessionInfo objects.
        """
        sessions: List[SessionInfo] = []

        if not os.path.exists(dir_path):
            return sessions

        try:
            files = [f for f in os.listdir(dir_path) if f.endswith(".jsonl")]
            for f in files:
                fp = os.path.join(dir_path, f)
                info = SessionManager._build_session_info(fp)
                if info:
                    sessions.append(info)
        except OSError:
            pass

        # Sort by modified time (newest first)
        sessions.sort(key=lambda s: s.modified.timestamp(), reverse=True)
        return sessions

    @staticmethod
    def _build_session_info(file_path: str) -> Optional[SessionInfo]:
        """Build SessionInfo from a session file.

        Args:
            file_path: Session file path.

        Returns:
            SessionInfo or None if file is invalid.
        """
        try:
            entries = load_entries_from_file(file_path)
            if len(entries) == 0:
                return None

            header = None
            for e in entries:
                if isinstance(e, SessionHeader):
                    header = e
                    break

            if header is None:
                return None

            stats_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            message_count = 0
            first_message = ""
            all_messages: List[str] = []
            name: Optional[str] = None

            for entry in entries:
                if isinstance(entry, SessionInfoEntry):
                    name = entry.name.strip() if entry.name else None

                if isinstance(entry, SessionMessageEntry):
                    message_count += 1
                    msg = entry.message
                    role = msg.get("role")
                    content = msg.get("content")

                    if role in ("user", "assistant"):
                        if isinstance(content, str):
                            text = content
                        elif isinstance(content, list):
                            text = " ".join(
                                b.get("text", "")
                                for b in content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        else:
                            text = ""

                        if text:
                            all_messages.append(text)
                            if not first_message and role == "user":
                                first_message = text

            # Determine modified time from last activity
            last_activity = SessionManager._get_last_activity_time(entries)
            if last_activity:
                modified = datetime.fromtimestamp(last_activity / 1000)
            else:
                try:
                    modified = datetime.fromisoformat(header.created_at)
                except (ValueError, TypeError):
                    modified = stats_mtime

            return SessionInfo(
                path=file_path,
                id=header.session_id,
                cwd=header.cwd,
                name=name,
                parent_session_path=header.parent_session,
                created=datetime.fromtimestamp(header.created_at / 1000),
                modified=modified,
                message_count=message_count,
                first_message=first_message or "(no messages)",
                all_messages_text=" ".join(all_messages),
            )
        except Exception:
            return None

    @staticmethod
    def _get_last_activity_time(entries: List[FileEntry]) -> Optional[int]:
        """Get timestamp of last user/assistant message activity.

        Args:
            entries: List of file entries.

        Returns:
            Unix timestamp in milliseconds, or None.
        """
        last_time: Optional[int] = None

        for entry in entries:
            if not isinstance(entry, SessionMessageEntry):
                continue

            msg = entry.message
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue

            # Check message timestamp
            msg_timestamp = msg.get("timestamp")
            if isinstance(msg_timestamp, (int, float)):
                last_time = max(last_time or 0, int(msg_timestamp))
                continue

            # Check entry timestamp
            entry_ts = entry.timestamp
            if isinstance(entry_ts, str):
                try:
                    t = int(datetime.fromisoformat(entry_ts).timestamp() * 1000)
                    last_time = max(last_time or 0, t)
                except ValueError:
                    pass

        return last_time