"""Slash-command completion.

WordCompleter offers its candidates for whatever word is under the cursor, so
typing an ordinary task and pressing space popped the command list up mid
sentence. This only completes when the *whole line* is a single token starting
with "/" -- which is the only moment a command could be what you meant.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document


class SlashCompleter(Completer):
    def __init__(self, commands: Mapping[str, str]):
        self._commands = dict(commands)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor

        # Only at the very start of the line, and only while still typing the
        # command itself. "/model gemini" is past the command, and "fix /tmp"
        # is a path, not a command.
        if not text.startswith("/") or " " in text:
            return

        for name, description in self._commands.items():
            if name.startswith(text):
                yield Completion(
                    name,
                    start_position=-len(text),
                    display=name,
                    display_meta=description,
                )
