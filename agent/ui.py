"""Terminal rendering.

Turns the loop's event stream into something readable. The loop itself stays
UI-free -- it emits events and this decides how they look, so the benchmark
adapter can run the identical loop with no console attached at all.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rich import box as rich_box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .tools import parse_arguments

# Windows consoles still default to cp1252, which cannot encode any of the box
# or arrow glyphs -- printing one raises UnicodeEncodeError and takes the whole
# session down. Detect it once and fall back to ASCII rather than crash.
_UNICODE = {
    "tick": "✓", "cross": "✗", "circle": "○", "bullet": "•",
    "arrow": "→", "prompt": "›", "dash": "—", "dot": "·", "ellipsis": "…",
}
_ASCII = {
    "tick": "+", "cross": "x", "circle": "o", "bullet": "*",
    "arrow": "->", "prompt": ">", "dash": "-", "dot": "-", "ellipsis": "...",
}
_glyphs: dict[str, str] | None = None


def ascii_only() -> bool:
    """True when stdout cannot carry the decorative glyphs.

    Resolved lazily: stdout's encoding is fixed up at startup, after import.
    """
    global _glyphs
    if _glyphs is None:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            "".join(_UNICODE.values()).encode(encoding)
            _glyphs = _UNICODE
        except (UnicodeEncodeError, LookupError):
            _glyphs = _ASCII
    return _glyphs is _ASCII


def glyph(name: str) -> str:
    ascii_only()  # populates the table
    assert _glyphs is not None
    return _glyphs[name]


def use_utf8_stdout() -> None:
    """Ask for UTF-8 before anything is printed. Harmless where it already is."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass

# How much of a tool result to show inline before collapsing it. Full output
# still goes to the model -- this is only what the human sees.
MAX_RESULT_LINES = 14
MAX_RESULT_WIDTH = 2000


def describe_tool_call(name: str, arguments: Any) -> tuple[str, str]:
    """Render a call as the thing it does, not as JSON.

    `run_shell {"command":"ls -l","timeout":30}` is noise; `$ ls -l` is the
    information.
    """
    args, err = parse_arguments(arguments)
    if err or args is None:
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments, default=str)
        return name, raw[:200]

    if name == "run_shell":
        command = str(args.get("command", ""))
        timeout = args.get("timeout")
        suffix = f"   (timeout {timeout}s)" if timeout not in (None, "", 30, "30") else ""
        return "shell", f"$ {command}{suffix}"

    if name == "write_file":
        content = args.get("content", "")
        content = content if isinstance(content, str) else str(content)
        lines = content.count("\n") + 1 if content else 0
        return "write", f"{args.get('path', '?')}   ({lines} lines, {len(content)} bytes)"

    if name == "read_file":
        return "read", str(args.get("path", "?"))

    if name == "task_complete":
        return "done", str(args.get("summary", ""))

    return name, json.dumps(args, default=str)[:200]


def collapse(output: str, max_lines: int = MAX_RESULT_LINES) -> str:
    """Trim a tool result for display, keeping the head and the tail -- errors
    tend to land at the end, so a head-only cut hides the useful part."""
    if len(output) > MAX_RESULT_WIDTH:
        output = output[:MAX_RESULT_WIDTH] + " …"
    lines = output.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = lines[: max_lines - 4]
    tail = lines[-3:]
    hidden = len(lines) - len(head) - len(tail)
    ell = glyph("ellipsis")
    return "\n".join([*head, f"{ell} {hidden} more lines {ell}", *tail])


class Renderer:
    """Consumes loop events and draws them."""

    def __init__(self, console: Console, show_steps: bool = False):
        self.console = console
        self.show_steps = show_steps
        self._status: Any = None

    # -- spinner ------------------------------------------------------------

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _start_status(self, label: str) -> None:
        self._stop_status()
        self._status = self.console.status(f"[dim]{label}[/dim]", spinner="dots")
        self._status.start()

    def close(self) -> None:
        self._stop_status()

    # -- events -------------------------------------------------------------

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        handler = getattr(self, f"_on_{event}", None)
        if handler is not None:
            handler(payload)

    def _on_step_start(self, payload: dict[str, Any]) -> None:
        if self.show_steps:
            self._stop_status()
            dash = glyph("dash")
            self.console.print(
                f"[dim]{dash} step {payload['step']}/{payload['max_steps']} {dash}[/dim]"
            )
        self._start_status("thinking")

    def _on_assistant_text(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        text = payload["text"].strip()
        if text:
            self.console.print(Markdown(text))

    def _on_recovered_tool_calls(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        self.console.print(
            f"[yellow]![/yellow] [dim]model wrote {payload['count']} tool call(s) as "
            f"text; recovered[/dim]"
        )

    def _on_tool_call(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        label, detail = describe_tool_call(payload["name"], payload["arguments"])
        if payload["name"] == "task_complete":
            return  # rendered by _on_complete instead
        marker = glyph("bullet")
        self.console.print(
            f"[bold cyan]{marker}[/bold cyan] [bold]{label}[/bold]  [dim]{detail}[/dim]"
        )
        self._start_status("running")

    def _on_tool_result(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        output = payload["output"]
        failed = output.startswith("Error:") or output.startswith("exit_code: ") and not output.startswith("exit_code: 0")
        body = collapse(output)
        if not body.strip():
            return
        style = "red" if failed else "dim"
        for line in body.splitlines():
            self.console.print(f"  [{style}]{line}[/{style}]")

    def _on_completion_deferred(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        self.console.print(
            "[yellow]![/yellow] [dim]claimed done in the same turn as the work; "
            "asked it to check the results first[/dim]"
        )

    def _on_complete(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        summary = payload.get("summary") or "done"
        self.console.print(f"\n[green]{glyph('tick')}[/green] {summary}")

    def _on_stopped(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        # Answering a question in prose is a perfectly good outcome -- the reply
        # has already been printed. Only a silent stop is worth flagging.
        if not payload.get("text", "").strip():
            self.console.print(
                f"\n[yellow]{glyph('circle')} stopped without doing anything[/yellow]"
            )

    def _on_max_iterations(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        self.console.print(
            f"\n[red]{glyph('cross')} hit the {payload['step']}-step limit[/red] "
            f"[dim](raise it with --max-iterations)[/dim]"
        )

    def _on_error(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        self.console.print(f"\n[red]{glyph('cross')} {payload['message']}[/red]")


def banner(
    console: Console,
    model: str,
    sandbox: str,
    mounts: list[tuple[str, str]],
    local: bool,
) -> None:
    lines = [Text.from_markup(f"[bold]cli-agent[/bold]  [dim]{model}[/dim]")]

    if local:
        lines.append(
            Text.from_markup("[red]running on your machine, unsandboxed[/red]")
        )
    else:
        lines.append(Text.from_markup(f"[dim]sandbox {sandbox}[/dim]"))

    if mounts:
        for host, target in mounts:
            lines.append(
                Text.from_markup(
                    f"[green]{target}[/green] [dim]{glyph('arrow')}[/dim] {host}"
                )
            )
        lines.append(Text.from_markup("[dim]files written there persist[/dim]"))
    elif not local:
        # They have lost files to this twice. Say it before the first turn.
        lines.append(
            Text.from_markup(
                "[yellow]no folder mounted — files are discarded on exit[/yellow]"
            )
        )
        lines.append(Text.from_markup("[dim]restart with --mount DIR to keep them[/dim]"))

    # Don't rely on rich's terminal detection for the border: we already know
    # whether this console can carry box-drawing characters.
    console.print(
        Panel(
            Group(*lines),
            border_style="dim",
            padding=(0, 2),
            box=rich_box.ASCII if ascii_only() else rich_box.ROUNDED,
        )
    )
    dot = glyph("dot")
    console.print(
        f"[dim]/help for commands {dot} Ctrl+C interrupts a turn {dot} /exit to quit"
        f"[/dim]\n"
    )


def turn_footer(console: Console, result: Any, elapsed: float) -> None:
    m = result.metrics()
    bits = [
        f"{m['steps']} steps",
        f"{m['tool_calls']} tools",
        f"{m.get('total_tokens', 0):,} tokens",
        f"{elapsed:.1f}s",
    ]
    if m["tool_errors"]:
        bits.append(f"[red]{m['tool_errors']} errors[/red]")
    if m["recovered_tool_calls"]:
        bits.append(f"[yellow]{m['recovered_tool_calls']} recovered[/yellow]")
    separator = " %s " % glyph("dot")
    console.print("[dim]%s[/dim]\n" % separator.join(bits))
