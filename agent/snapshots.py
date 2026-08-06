"""Checkpoints taken before the agent changes a file.

The permission gate asks whether an edit should happen. A snapshot answers the
next question -- what if it should not have -- and that is what makes the looser
gate modes survivable: with a copy of every file as it was before the agent
touched it, "accept edits" and "auto" stop being one-way doors.

Implemented as an Executor wrapper for the same reason the gate is: it sits
above both LocalExecutor and DockerExecutor, so a container write is captured
by exactly the same code as a host write, and nothing in tools.py has to know.

Copies live beside the transcripts in ~/.dietcode, never in the project -- a
snapshot directory inside the repo is a snapshot directory that gets committed,
and these are copies of files that may hold anything.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sandbox import DEFAULT_TIMEOUT, Executor, SandboxError, ShellResult
from .sessions import project_dir

# A snapshot is only useful if taking it is cheap enough to never think about.
# Past this size the copy costs more than the safety is worth, and the file is
# almost certainly not hand-written source.
MAX_SNAPSHOT_BYTES = 2_000_000


@dataclass
class Change:
    """One file, as it was before a tool changed it."""

    index: int
    path: str
    tool: str
    at: float
    existed: bool
    backup: Path | None = None

    @property
    def label(self) -> str:
        return f"{self.tool} {self.path}" + ("" if self.existed else "  (new file)")


class SnapshotStore:
    """Keeps the pre-edit copies for one session."""

    def __init__(self, project: str | Path, session_id: str, enabled: bool = True):
        self.enabled = enabled
        self.session_id = session_id
        self.root = project_dir(project) / "snapshots" / session_id
        self.changes: list[Change] = []

    def _backup_path(self, index: int, path: str) -> Path:
        # Flat directory, index-prefixed: two files with the same basename in
        # different directories must not collide.
        safe = Path(path).name or "file"
        return self.root / f"{index:04d}-{safe}"

    def capture(self, path: str, tool: str, before: str | None) -> Change | None:
        """Record a file's contents before it is changed.

        `before` is None when the file did not exist -- worth recording too,
        because undoing a create means deleting it.
        """
        if not self.enabled:
            return None

        index = len(self.changes) + 1
        change = Change(
            index=index, path=path, tool=tool, at=time.time(), existed=before is not None
        )
        if before is not None:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                backup = self._backup_path(index, path)
                backup.write_text(before, encoding="utf-8")
                change.backup = backup
            except OSError:
                # A snapshot that cannot be written must not stop the edit.
                # The user asked for the edit; the checkpoint is the extra.
                self.enabled = False
                return None
        self.changes.append(change)
        self._write_index()
        return change

    def _write_index(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "index": c.index,
                            "path": c.path,
                            "tool": c.tool,
                            "at": c.at,
                            "existed": c.existed,
                            "backup": str(c.backup) if c.backup else None,
                        }
                        for c in self.changes
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    # -- restoring ----------------------------------------------------------

    def restore(self, executor: Executor, change: Change) -> str:
        """Put one file back. Returns a line describing what happened.

        Writes through the *unwrapped* executor. Restoring through the
        snapshotting layer would checkpoint the restore itself, so undo_all
        would put a change back for every change it took off and never finish.
        """
        executor = _unwrap(executor)
        if not change.existed:
            # Undoing a create. Removed through the executor so it works in a
            # container as well as on the host.
            result = executor.run_shell(f"rm -f -- {_quote(change.path)}")
            if result.exit_code != 0:
                return f"could not remove {change.path}: {result.stderr.strip()}"
            return f"removed {change.path}"

        if change.backup is None or not change.backup.exists():
            return f"no saved copy of {change.path}"
        try:
            contents = change.backup.read_text(encoding="utf-8")
        except OSError as exc:
            return f"could not read the saved copy of {change.path}: {exc}"

        try:
            executor.write_file(change.path, contents)
        except SandboxError as exc:
            return f"could not restore {change.path}: {exc}"
        return f"restored {change.path}"

    def undo_last(self, executor: Executor) -> list[str]:
        """Roll back the most recent change."""
        if not self.changes:
            return []
        change = self.changes.pop()
        self._write_index()
        return [self.restore(executor, change)]

    def undo_all(self, executor: Executor) -> list[str]:
        """Roll back everything this session touched, newest first.

        Order matters: a file written twice must end up at its oldest saved
        state, and walking backwards gets there.
        """
        lines = []
        while self.changes:
            lines.extend(self.undo_last(executor))
        return lines


def _quote(path: str) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def _unwrap(executor: Any) -> Any:
    """Peel off any snapshotting layers, so an undo is not itself snapshotted."""
    seen = 0
    while isinstance(executor, Snapshotting) and seen < 10:
        executor = executor._inner
        seen += 1
    return executor


class Snapshotting:
    """Executor wrapper that checkpoints a file before every change to it."""

    def __init__(self, inner: Executor, store: SnapshotStore):
        self._inner = inner
        self.store = store

    def __getattr__(self, name: str) -> Any:
        # Anything not intercepted -- close(), container, workdir, root --
        # belongs to the wrapped executor.
        return getattr(self._inner, name)

    def _before(self, path: str) -> str | None:
        try:
            contents = self._inner.read_file(path)
        except (SandboxError, OSError):
            return None  # did not exist, or is unreadable
        if len(contents) > MAX_SNAPSHOT_BYTES:
            return None
        return contents

    def write_file(self, path: str, content: str) -> str:
        self.store.capture(path, "write", self._before(path))
        return self._inner.write_file(path, content)

    # No edit_file here on purpose: no Executor implements one. `edit_file` the
    # *tool* reads, substitutes and writes back through write_file, so both
    # tools are covered by the single interception above.

    def run_shell(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> ShellResult:
        # Deliberately not snapshotted. A shell command can touch anything, and
        # guessing which paths it will write is worse than being honest that
        # undo covers file tools only.
        return self._inner.run_shell(command, timeout=timeout)
