"""Interactive session.

One container and one conversation for the whole session, so the agent
remembers what it already did and the filesystem it built is still there on the
next turn. The one-shot CLI path throws both away between tasks.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from .loop import DEFAULT_CONTEXT_BUDGET, DEFAULT_MAX_ITERATIONS, agent_loop
from .sandbox import SandboxError
from .ui import (
    BRAND,
    MUTED,
    NOTE,
    OUTPUT,
    TOOL,
    WARN,
    Renderer,
    banner,
    glyph,
    turn_footer,
)

COMMANDS = {
    "/help": "show this help",
    "/clear": "forget the conversation (the sandbox and its files stay)",
    "/files": "list files in the working directory",
    "/cost": "tokens used so far this session",
    "/sandbox": "show the container and any mounted folders",
    "/exit": "quit (also Ctrl+D)",
}


def _read_input(console: Console) -> str:
    """Prompt with history and arrow-key editing where possible.

    Falls back to input() when prompt_toolkit is missing or stdin is not a
    terminal -- piping a script in must still work, and prompt_toolkit raises
    on a non-tty rather than degrading.
    """
    marker = glyph("prompt")
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.formatted_text import ANSI
            from prompt_toolkit.history import InMemoryHistory

            if not hasattr(_read_input, "_session"):
                _read_input._session = PromptSession(history=InMemoryHistory())  # type: ignore[attr-defined]
            # prompt_toolkit does its own rendering, so the colour is a raw
            # escape rather than rich markup.
            return _read_input._session.prompt(  # type: ignore[attr-defined]
                ANSI(f"\x1b[1;91m{marker}\x1b[0m ")
            )
        except ImportError:
            pass
    console.print(f"[{BRAND}]{marker}[/{BRAND}] ", end="")
    return input()


class Session:
    def __init__(
        self,
        executor: Any,
        client: Any,
        model: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        mounts: list[tuple[str, str]] | None = None,
        local: bool = False,
        show_steps: bool = False,
        stream: bool = True,
        context_budget: int = DEFAULT_CONTEXT_BUDGET,
        max_total_tokens: int | None = None,
    ):
        self.context_budget = context_budget
        self.max_total_tokens = max_total_tokens
        self.executor = executor
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.mounts = mounts or []
        self.local = local
        self.stream = stream
        self.console = Console()
        self.renderer = Renderer(self.console, show_steps=show_steps)
        self.history: list[dict[str, Any]] | None = None
        self.total_tokens = 0
        self.total_steps = 0
        self.turns = 0

    # -- slash commands -----------------------------------------------------

    def _cmd_help(self) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        for name, description in COMMANDS.items():
            table.add_row(
                f"[{TOOL}]{name}[/{TOOL}]", f"[{MUTED}]{description}[/{MUTED}]"
            )
        self.console.print(table)
        self.console.print(
            f"[{MUTED}]Anything else is a task. The agent keeps its memory and its "
            f"files between turns.[/{MUTED}]\n"
        )

    def _cmd_clear(self) -> None:
        self.history = None
        self.console.print(
            f"[{NOTE}]conversation cleared; files and container kept[/{NOTE}]\n"
        )

    def _cmd_files(self) -> None:
        try:
            result = self.executor.run_shell("ls -la")
        except SandboxError as exc:
            self.console.print(f"[red1]{exc}[/red1]\n")
            return
        self.console.print(
            f"[{OUTPUT}]{result.stdout.strip() or '(empty)'}[/{OUTPUT}]\n"
        )

    def _cmd_cost(self) -> None:
        dot = glyph("dot")
        self.console.print(
            f"[{NOTE}]{self.turns} turns {dot} {self.total_steps} steps {dot} "
            f"{self.total_tokens:,} tokens[/{NOTE}]\n"
        )

    def _cmd_sandbox(self) -> None:
        if self.local:
            self.console.print(f"[{WARN}]running unsandboxed on the host[/{WARN}]\n")
            return
        self.console.print(f"[{MUTED}]container {self.executor.container}[/{MUTED}]")
        if self.mounts:
            for host, target in self.mounts:
                self.console.print(
                    f"[{NOTE}]{target}[/{NOTE}] [{MUTED}]{glyph('arrow')} {host} "
                    f"(persists)[/{MUTED}]"
                )
        else:
            self.console.print(
                f"[{WARN}]nothing mounted {glyph('dash')} files vanish when you "
                f"exit[/{WARN}]"
            )
        self.console.print()

    def handle_command(self, text: str) -> bool:
        """Returns False when the session should end."""
        command = text.strip().split()[0].lower()
        if command in ("/exit", "/quit", "/q"):
            return False
        handlers = {
            "/help": self._cmd_help,
            "/?": self._cmd_help,
            "/clear": self._cmd_clear,
            "/files": self._cmd_files,
            "/cost": self._cmd_cost,
            "/sandbox": self._cmd_sandbox,
        }
        handler = handlers.get(command)
        if handler is None:
            self.console.print(
                f"[red1]unknown command {command}[/red1] "
                f"[{MUTED}]{glyph('dash')} /help for the list[/{MUTED}]\n"
            )
        else:
            handler()
        return True

    # -- main loop ----------------------------------------------------------

    def run_turn(self, task: str) -> None:
        started = time.monotonic()
        try:
            result = agent_loop(
                task,
                self.executor,
                client=self.client,
                model=self.model,
                max_iterations=self.max_iterations,
                stream=self.stream,
                context_budget=self.context_budget,
                max_total_tokens=self.max_total_tokens,
                history=self.history,
                on_event=self.renderer.on_event,
            )
        except KeyboardInterrupt:
            # The turn is discarded rather than half-kept: a transcript with
            # tool calls that were never answered is rejected by the API on the
            # next request.
            self.renderer.close()
            self.console.print(
                f"\n[{WARN}]interrupted {glyph('dash')} turn discarded[/{WARN}]\n"
            )
            return
        finally:
            self.renderer.close()

        self.history = result.messages
        self.total_tokens += result.usage.get("total_tokens", 0)
        self.total_steps += result.steps
        self.turns += 1
        turn_footer(self.console, result, time.monotonic() - started)

    def run(self) -> int:
        banner(
            self.console,
            self.model,
            getattr(self.executor, "container", "local"),
            self.mounts,
            self.local,
        )

        while True:
            try:
                text = _read_input(self.console)
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break

            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if not self.handle_command(text):
                    break
                continue

            self.run_turn(text)

        if self.turns:
            self.console.print(
                f"[{MUTED}]{self.turns} turns {glyph('dot')} {self.total_tokens:,} "
                f"tokens this session[/{MUTED}]"
            )
        if self.mounts:
            for host, _target in self.mounts:
                self.console.print(f"[{NOTE}]your files are in {host}[/{NOTE}]")
        elif not self.local:
            self.console.print(
                f"[{WARN}]files from this session were discarded with the "
                f"container[/{WARN}]"
            )
        return 0
