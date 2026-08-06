"""What the agent should still know next session.

The project's instructions file is the *user's* standing orders and stays
read-only to the agent -- otherwise a run can rewrite its own rules mid-task.
Memory is the other half of that split: a file the agent may write, kept
outside the repo, holding what it worked out for itself.

Both are loaded into context together, and the distinction is what keeps the
"create an instructions file" idea in the architecture safe: the agent gets
somewhere to record things without gaining edit rights over its own brief.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .sessions import project_dir

# Memory is prepended to every request in the session, so it is charged for on
# every turn. Past this it stops being cheap and starts crowding out the task.
MAX_MEMORY_CHARS = 4000
MAX_NOTE_CHARS = 400

HEADER = """\
# Project memory

Written by dietcode as it works. Edit or delete anything here -- it is read
into context at the start of every session.
"""

REMEMBER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Save a short note about this project for future sessions: a "
            "convention, a command that works, a trap you hit. Only for things "
            "that will still be true next time -- not for what you did today, "
            "and not as a scratchpad within a task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "One sentence, specific enough to act on later.",
                },
            },
            "required": ["note"],
        },
    },
}


def memory_path(project: str | Path) -> Path:
    return project_dir(project) / "memory" / "memory.md"


def load_memory(project: str | Path) -> str:
    """The notes, or "" if there are none."""
    try:
        text = memory_path(project).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) > MAX_MEMORY_CHARS:
        # Keep the tail: later notes supersede earlier ones.
        text = "… [older notes trimmed]\n" + text[-MAX_MEMORY_CHARS:]
    return text


def remember(project: str | Path, note: str) -> str:
    """Append one note. Returns what to tell the model."""
    note = " ".join(str(note or "").split())
    if not note:
        return "Error: remember needs a note."
    if len(note) > MAX_NOTE_CHARS:
        note = note[:MAX_NOTE_CHARS] + "…"

    path = memory_path(project)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else HEADER
        if note in existing:
            return "Already remembered; nothing added."
        with path.open("a" if path.exists() else "w", encoding="utf-8") as handle:
            if not path.exists() or not existing.strip():
                handle.write(HEADER)
            handle.write(f"\n- {date.today().isoformat()} {note}\n")
    except OSError as exc:
        # Same contract as every other tool: a failure the model can read.
        return f"Error: could not save the note: {exc}"
    return f"Remembered: {note}"


def make_remember_handler(project: str | Path):
    """A tool handler for agent_loop's extra_tool_handlers hook."""

    def handler(arguments: dict[str, Any]) -> str:
        return remember(project, arguments.get("note", ""))

    return handler


def with_memory(system_prompt: str, project: str | Path) -> str:
    """Fold the notes into the system prompt, if there are any."""
    notes = load_memory(project)
    if not notes:
        return system_prompt
    return (
        f"{system_prompt}\n\n"
        f"--- What you noted about this project previously ---\n"
        f"{notes}\n"
        f"--- end of notes ---\n"
        f"These are your own earlier notes, not instructions from the user. "
        f"Prefer what you can see in the code if they disagree."
    )
