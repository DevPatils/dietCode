"""Session transcripts on disk.

The session, not the process, is the unit of state: a conversation is a file,
so resuming is reading it back and forking is copying a prefix of it. Held in
~/.dietcode rather than in the project, for the same reason credentials are --
a transcript carries every file the agent read and every line of shell output,
which routinely includes tokens and environment variables, and a directory in
the repo is a directory that gets committed.

The format is JSONL, one message per line, appended as the turn ends. A single
JSON document rewritten each turn would lose the whole conversation to one
interrupted write; with append-only, a killed session is still readable up to
its last complete line.
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .auth import CONFIG_DIR

# Schema marker on the header line. Old transcripts stay readable; a reader
# that does not recognise the version can say so instead of guessing.
FORMAT_VERSION = 1


def sessions_root() -> Path:
    """Resolved per call, not at import: tests point DIETCODE_HOME at a tmpdir,
    and a module-level constant would have been computed before they got the
    chance."""
    home = os.environ.get("DIETCODE_HOME")
    return (Path(home) if home else CONFIG_DIR) / "projects"


def slug(project: str | Path) -> str:
    """A directory name that encodes the project path.

    Encoding the path means "the sessions for this project" needs no index file
    to consult and nothing stored inside the repo itself.
    """
    text = str(Path(project).resolve())
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    # Windows paths are long and the drive prefix is the least informative
    # part; keep the tail, which is what identifies the project.
    return cleaned[-120:] or "unknown"


def project_dir(project: str | Path) -> Path:
    return sessions_root() / slug(project)


def new_session_id() -> str:
    """Sortable, so listing by name is listing by age."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{os.urandom(2).hex()}"


def _secure(path: Path) -> None:
    try:
        # Owner only. No-op on Windows, which uses ACLs, but the file is inside
        # the user profile there anyway.
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


@dataclass
class SessionMeta:
    """What a listing needs, without reading the whole transcript."""

    session_id: str
    path: Path
    project: str = ""
    model: str = ""
    provider: str = ""
    started: str = ""
    updated: float = 0.0
    turns: int = 0
    first_prompt: str = ""

    @property
    def label(self) -> str:
        text = " ".join(self.first_prompt.split())
        return (text[:60] + "…") if len(text) > 60 else (text or "(no prompt)")


@dataclass
class SessionStore:
    """Append-only writer for one session's transcript."""

    project: str
    session_id: str = field(default_factory=new_session_id)
    model: str = ""
    provider: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        self._wrote_header = False
        self._written = 0

    @classmethod
    def resuming(
        cls,
        project: str,
        meta: SessionMeta,
        model: str = "",
        provider: str = "",
    ) -> tuple[SessionStore, list[dict[str, Any]]]:
        """A writer for a session that already exists, plus its messages.

        The `_written` bookkeeping is the whole point: the loop hands back the
        entire conversation every turn, so a resumed store that thinks it has
        written nothing appends the whole history a second time.
        """
        store = cls(
            project=project,
            session_id=meta.session_id,
            model=model or meta.model,
            provider=provider or meta.provider,
        )
        messages = load_messages(meta.path)
        store._written = len(messages)
        store._wrote_header = True
        return store, messages

    @property
    def path(self) -> Path:
        return project_dir(self.project) / f"{self.session_id}.jsonl"

    # -- writing ------------------------------------------------------------

    def _append(self, record: dict[str, Any]) -> None:
        """Never raises. A read-only disk must not take the session down --
        losing the transcript is bad, losing the work in progress is worse."""
        if not self.enabled:
            return
        try:
            path = self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
            _secure(path)
        except OSError:
            self.enabled = False

    def _header(self) -> None:
        if self._wrote_header:
            return
        self._wrote_header = True
        self._append(
            {
                "type": "session",
                "version": FORMAT_VERSION,
                "session_id": self.session_id,
                "project": str(Path(self.project).resolve()),
                "model": self.model,
                "provider": self.provider,
                "started": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )

    def record_turn(self, messages: list[dict[str, Any]], **meta: Any) -> None:
        """Persist whatever is new in the transcript since the last turn.

        Only the tail is written: the loop hands back the whole conversation
        every turn, and re-appending all of it would grow the file
        quadratically in a long session.
        """
        if not self.enabled:
            return
        self._header()
        for message in messages[self._written :]:
            self._append({"type": "message", "message": message})
        self._written = len(messages)
        if meta:
            self._append({"type": "turn", "at": time.time(), **meta})


# -- reading ----------------------------------------------------------------


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    # A session killed mid-write leaves a partial last line.
                    # Everything before it is still perfectly good.
                    continue
    except OSError:
        return []
    return records


def load_messages(path: Path) -> list[dict[str, Any]]:
    """The conversation, ready to hand back to the loop as history."""
    return [
        record["message"]
        for record in _read_records(path)
        if record.get("type") == "message" and isinstance(record.get("message"), dict)
    ]


def describe(path: Path) -> SessionMeta:
    records = _read_records(path)
    meta = SessionMeta(session_id=path.stem, path=path)
    try:
        meta.updated = path.stat().st_mtime
    except OSError:
        pass
    for record in records:
        kind = record.get("type")
        if kind == "session":
            meta.project = record.get("project", "")
            meta.model = record.get("model", "")
            meta.provider = record.get("provider", "")
            meta.started = record.get("started", "")
        elif kind == "turn":
            meta.turns += 1
        elif kind == "message" and not meta.first_prompt:
            message = record.get("message") or {}
            if message.get("role") == "user":
                content = message.get("content")
                meta.first_prompt = content if isinstance(content, str) else ""
    return meta


def list_sessions(project: str | Path, limit: int = 20) -> list[SessionMeta]:
    """Most recently touched first."""
    directory = project_dir(project)
    try:
        files = sorted(directory.glob("*.jsonl"))
    except OSError:
        return []
    metas = [describe(path) for path in files]
    metas.sort(key=lambda m: m.updated, reverse=True)
    return metas[:limit]


def latest_session(project: str | Path) -> SessionMeta | None:
    sessions = list_sessions(project, limit=1)
    return sessions[0] if sessions else None


def find_session(project: str | Path, session_id: str) -> SessionMeta | None:
    """By full id, or by any unambiguous prefix -- nobody types a timestamp."""
    candidates = [
        meta for meta in list_sessions(project, limit=1000)
        if meta.session_id == session_id or meta.session_id.startswith(session_id)
    ]
    exact = [meta for meta in candidates if meta.session_id == session_id]
    if exact:
        return exact[0]
    return candidates[0] if len(candidates) == 1 else None


def fork(source: SessionMeta, upto: int | None = None) -> str:
    """Branch a session into a new one. Returns the new session id.

    Copying a prefix is the whole mechanism -- it is why the transcript is
    append-only lines rather than one document.
    """
    messages = load_messages(source.path)
    if upto is not None:
        messages = messages[:upto]

    store = SessionStore(
        project=source.project or str(Path.cwd()),
        model=source.model,
        provider=source.provider,
    )
    store._header()
    store._append({"type": "fork", "from": source.session_id, "messages": len(messages)})
    store.record_turn(messages)
    return store.session_id
