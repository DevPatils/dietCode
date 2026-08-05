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

from .auth import (
    PROVIDERS,
    AuthError,
    default_provider,
    get_provider,
    resolve_key,
)
from .commands import auth_status, doctor, login, logout
from .loop import (
    DEFAULT_CONTEXT_BUDGET,
    DEFAULT_MAX_ITERATIONS,
    agent_loop,
    estimate_tokens,
    make_client,
)
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
    context_percent,
    glyph,
    input_rule,
    sandbox_label,
    status_bar,
    turn_footer,
)

COMMANDS = {
    "/help": "show this help",
    "/login": "save an API key, without leaving the session",
    "/logout": "forget a stored key",
    "/auth": "which providers are usable",
    "/provider": "switch provider, e.g. /provider gemini",
    "/model": "switch model, e.g. /model llama-3.1-8b-instant",
    "/clear": "forget the conversation (the sandbox and its files stay)",
    "/files": "list files in the working directory",
    "/cost": "tokens used so far this session",
    "/sandbox": "show the container and any mounted folders",
    "/doctor": "check this machine is set up correctly",
    "/exit": "quit (also Ctrl+D)",
}


PLACEHOLDERS = [
    'Try "add a test for the parser and make it pass"',
    'Try "find why the build is failing and fix it"',
    'Try "write a script that dedupes this csv, then run it"',
]


def _read_input(
    console: Console, status: str | None = None, placeholder: str | None = None
) -> str:
    """Prompt with history, arrow-key editing, and a pinned status bar.

    Falls back to input() when prompt_toolkit is missing or stdin is not a
    terminal -- piping a script in must still work, and prompt_toolkit raises
    on a non-tty rather than degrading.
    """
    marker = glyph("prompt")
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.formatted_text import ANSI
            from prompt_toolkit.history import InMemoryHistory

            if not hasattr(_read_input, "_session"):
                # Only slash commands complete: every candidate starts with "/",
                # so ordinary prose never triggers a suggestion popup.
                _read_input._session = PromptSession(  # type: ignore[attr-defined]
                    history=InMemoryHistory(),
                    completer=WordCompleter(list(COMMANDS), WORD=True),
                )
            # prompt_toolkit renders these itself, so they are raw escapes
            # rather than rich markup.
            return _read_input._session.prompt(  # type: ignore[attr-defined]
                ANSI(f"\x1b[1;91m{marker}\x1b[0m "),
                bottom_toolbar=ANSI(status) if status else None,
                placeholder=ANSI(f"\x1b[38;5;240m{placeholder}\x1b[0m")
                if placeholder
                else None,
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
        provider: str | None = None,
        extras: dict[str, Any] | None = None,
    ):
        self.context_budget = context_budget
        self.max_total_tokens = max_total_tokens
        # Project instructions and sub-agent wiring, built once by the CLI.
        self.extras = extras or {}
        # Tracked so /login and /provider can rebuild the client in place.
        self.provider = provider or default_provider()
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

    def _status_bar(self) -> str:
        """The pinned line under the input: where, how isolated, what model."""
        if self.mounts:
            location = self.mounts[0][0]
        elif self.local:
            location = str(getattr(self.executor, "workdir", "."))
        else:
            location = getattr(self.executor, "workdir", "/workspace")
        location = str(location)
        if len(location) > 34:
            location = "..." + location[-31:]

        # Context left is the honest number here: it is what decides when old
        # turns start getting dropped.
        used = estimate_tokens(self.history) if self.history else 0
        return status_bar(
            location=location,
            sandbox=sandbox_label(
                getattr(self.executor, "container", None), self.mounts, self.local
            ),
            model=self.model,
            context_left=context_percent(used, self.context_budget),
            width=self.console.width,
        )

    # -- slash commands -----------------------------------------------------

    def _cmd_help(self, _args: list[str]) -> None:
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

    def _cmd_clear(self, _args: list[str]) -> None:
        self.history = None
        self.console.print(
            f"[{NOTE}]conversation cleared; files and container kept[/{NOTE}]\n"
        )

    def _cmd_files(self, _args: list[str]) -> None:
        try:
            result = self.executor.run_shell("ls -la")
        except SandboxError as exc:
            self.console.print(f"[red1]{exc}[/red1]\n")
            return
        self.console.print(
            f"[{OUTPUT}]{result.stdout.strip() or '(empty)'}[/{OUTPUT}]\n"
        )

    def _cmd_cost(self, _args: list[str]) -> None:
        dot = glyph("dot")
        self.console.print(
            f"[{NOTE}]{self.turns} turns {dot} {self.total_steps} steps {dot} "
            f"{self.total_tokens:,} tokens[/{NOTE}]\n"
        )

    def _cmd_sandbox(self, _args: list[str]) -> None:
        # getattr, not attribute access: the executor may be a host executor or
        # a PermissionGate wrapping one, and neither has a container.
        container = getattr(self.executor, "container", None)
        if self.local or container is None:
            self.console.print(
                f"[{WARN}]no container {glyph('dash')} running directly on your "
                f"machine[/{WARN}]"
            )
            root = getattr(self.executor, "root", None) or getattr(
                self.executor, "workdir", "."
            )
            self.console.print(f"[{MUTED}]working in {root}[/{MUTED}]\n")
            return
        self.console.print(f"[{MUTED}]container {container}[/{MUTED}]")
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

    # -- account and model, without leaving the session ----------------------

    def _rebuild_client(self) -> bool:
        """Point the session at whatever provider is currently selected."""
        spec = get_provider(self.provider)
        api_key, _source = resolve_key(self.provider)
        if not api_key:
            self.console.print(
                f"[{WARN}]no key for {spec.label}[/{WARN}] "
                f"[{MUTED}]{glyph('dash')} run /login {spec.name}[/{MUTED}]\n"
            )
            return False
        self.client = make_client(api_key=api_key, base_url=spec.base_url)
        return True

    def _cmd_login(self, args: list[str]) -> None:
        login(self.console, args[0] if args else None, None)
        # Whichever provider was just saved becomes the one to use, so the very
        # next turn works without a restart.
        self.provider = default_provider()
        spec = get_provider(self.provider)
        self.model = spec.default_model
        if self._rebuild_client():
            self.console.print(
                f"[{NOTE}]now using {spec.label} {glyph('dot')} {self.model}[/{NOTE}]\n"
            )

    def _cmd_logout(self, args: list[str]) -> None:
        logout(self.console, args[0] if args else None)
        self.provider = default_provider()
        self._rebuild_client()
        self.console.print()

    def _cmd_auth(self, _args: list[str]) -> None:
        auth_status(self.console)
        self.console.print()

    def _cmd_doctor(self, _args: list[str]) -> None:
        doctor(self.console)
        self.console.print()

    def _cmd_provider(self, args: list[str]) -> None:
        if not args:
            names = ", ".join(PROVIDERS)
            self.console.print(
                f"[{MUTED}]currently {self.provider} {glyph('dot')} "
                f"available: {names}[/{MUTED}]\n"
            )
            return
        try:
            spec = get_provider(args[0])
        except AuthError as exc:
            self.console.print(f"[red1]{exc}[/red1]\n")
            return

        previous, previous_model = self.provider, self.model
        self.provider, self.model = spec.name, spec.default_model
        if not self._rebuild_client():
            self.provider, self.model = previous, previous_model
            return
        # The conversation carries over: switching model mid-task is the point,
        # and the transcript is provider-independent.
        self.console.print(
            f"[{NOTE}]switched to {spec.label} {glyph('dot')} {self.model}[/{NOTE}]\n"
        )

    def _cmd_model(self, args: list[str]) -> None:
        if not args:
            self.console.print(f"[{MUTED}]currently {self.model}[/{MUTED}]\n")
            return
        self.model = args[0]
        self.console.print(f"[{NOTE}]model set to {self.model}[/{NOTE}]\n")

    def handle_command(self, text: str) -> bool:
        """Returns False when the session should end."""
        parts = text.strip().split()
        command, args = parts[0].lower(), parts[1:]
        if command in ("/exit", "/quit", "/q"):
            return False
        handlers = {
            "/help": self._cmd_help,
            "/?": self._cmd_help,
            "/login": self._cmd_login,
            "/logout": self._cmd_logout,
            "/auth": self._cmd_auth,
            "/status": self._cmd_auth,
            "/provider": self._cmd_provider,
            "/model": self._cmd_model,
            "/doctor": self._cmd_doctor,
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
            handler(args)
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
                **self.extras,
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
            str(getattr(self.executor, "container", None) or getattr(self.executor, "root", ".")),
            self.mounts,
            self.local,
        )

        while True:
            input_rule(self.console)
            try:
                text = _read_input(
                    self.console,
                    status=self._status_bar(),
                    # Only on the first, empty prompt -- a suggestion after the
                    # conversation has started is noise.
                    placeholder=PLACEHOLDERS[0] if self.turns == 0 else None,
                )
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
