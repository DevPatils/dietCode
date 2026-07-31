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

from .loop import DEFAULT_MAX_ITERATIONS, agent_loop
from .sandbox import SandboxError
from .ui import Renderer, banner, glyph, turn_footer

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
            from prompt_toolkit.history import InMemoryHistory

            if not hasattr(_read_input, "_session"):
                _read_input._session = PromptSession(history=InMemoryHistory())  # type: ignore[attr-defined]
            return _read_input._session.prompt(f"{marker} ")  # type: ignore[attr-defined]
        except ImportError:
            pass
    console.print(f"[bold]{marker}[/bold] ", end="")
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
    ):
        self.executor = executor
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.mounts = mounts or []
        self.local = local
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
            table.add_row(f"[bold]{name}[/bold]", f"[dim]{description}[/dim]")
        self.console.print(table)
        self.console.print(
            "[dim]Anything else is a task. The agent keeps its memory and its "
            "files between turns.[/dim]\n"
        )

    def _cmd_clear(self) -> None:
        self.history = None
        self.console.print("[dim]conversation cleared; files and container kept[/dim]\n")

    def _cmd_files(self) -> None:
        try:
            result = self.executor.run_shell("ls -la")
        except SandboxError as exc:
            self.console.print(f"[red]{exc}[/red]\n")
            return
        self.console.print(f"[dim]{result.stdout.strip() or '(empty)'}[/dim]\n")

    def _cmd_cost(self) -> None:
        self.console.print(
            f"[dim]{self.turns} turns · {self.total_steps} steps · "
            f"{self.total_tokens:,} tokens[/dim]\n"
        )

    def _cmd_sandbox(self) -> None:
        if self.local:
            self.console.print("[yellow]running unsandboxed on the host[/yellow]\n")
            return
        self.console.print(f"[dim]container {self.executor.container}[/dim]")
        if self.mounts:
            for host, target in self.mounts:
                self.console.print(f"[dim]{target} → {host} (persists)[/dim]")
        else:
            self.console.print(
                "[yellow]nothing mounted — files vanish when you exit[/yellow]"
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
                f"[red]unknown command {command}[/red] [dim]— /help for the list[/dim]\n"
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
                history=self.history,
                on_event=self.renderer.on_event,
            )
        except KeyboardInterrupt:
            # The turn is discarded rather than half-kept: a transcript with
            # tool calls that were never answered is rejected by the API on the
            # next request.
            self.renderer.close()
            self.console.print("\n[yellow]interrupted — turn discarded[/yellow]\n")
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
                f"[dim]{self.turns} turns · {self.total_tokens:,} tokens this "
                f"session[/dim]"
            )
        if self.mounts:
            for host, _target in self.mounts:
                self.console.print(f"[dim]your files are in {host}[/dim]")
        elif not self.local:
            self.console.print(
                "[yellow]files from this session were discarded with the "
                "container[/yellow]"
            )
        return 0
