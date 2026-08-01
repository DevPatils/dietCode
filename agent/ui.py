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
    "tick": "✓", "cross": "✗", "circle": "○", "bullet": "▪",
    "arrow": "→", "prompt": "❯", "dash": "—", "dot": "·", "ellipsis": "…",
    "bar": "│",
}
_ASCII = {
    "tick": "+", "cross": "x", "circle": "o", "bullet": "*",
    "arrow": "->", "prompt": ">", "dash": "-", "dot": "-", "ellipsis": "...",
    "bar": "|",
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

# --- palette ----------------------------------------------------------------
#
# Red-forward. The catch with a red theme is that red is also the universal
# "something broke" signal, so if chrome and errors share it, failures stop
# registering. Resolved by reserving *reversed* red for anything that went
# wrong -- it reads as an alarm even surrounded by red -- and giving the rest a
# ramp from bright (live) to muted (background).
#
# Written as literal markup rather than a rich Theme so any Console renders it,
# including ones tests construct themselves.
BRAND = "bold bright_red"
BORDER = "red3"
MARKER = "bright_red"       # the bullet in front of a tool call
TOOL = "bold #ff8080"       # the tool's name
DETAIL = "#b06a6a"          # its arguments
OUTPUT = "grey42"           # tool output, deliberately recessive
NOTE = "#d75f5f"            # session chatter: mounts, costs, hints
WARN = "orange3"            # recovered / deferred / trimmed: odd, not fatal
OK = "bold bright_red"      # success; the tick carries the meaning
FAIL = "bold white on red3"  # reversed, so it survives a red background
MUTED = "grey37"


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
        self._streaming = False

    # -- spinner ------------------------------------------------------------

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _start_status(self, label: str) -> None:
        self._stop_status()
        self._status = self.console.status(
            f"[{DETAIL}]{label}[/{DETAIL}]", spinner="dots", spinner_style=MARKER
        )
        self._status.start()

    def close(self) -> None:
        self._end_stream()
        self._stop_status()

    # -- events -------------------------------------------------------------

    def _end_stream(self) -> None:
        """Close off a partial streamed line so the next output starts clean."""
        if self._streaming:
            self.console.print()
            self._streaming = False

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        if event != "assistant_delta":
            self._end_stream()
        handler = getattr(self, f"_on_{event}", None)
        if handler is not None:
            handler(payload)

    def _on_assistant_delta(self, payload: dict[str, Any]) -> None:
        text = payload.get("text", "")
        if not text:
            return
        self._stop_status()
        self._streaming = True
        # Raw, not markup: streamed fragments are arbitrary model text and a
        # stray '[' would otherwise be parsed as a rich style tag.
        self.console.print(text, end="", markup=False, highlight=False)

    def _on_step_start(self, payload: dict[str, Any]) -> None:
        if self.show_steps:
            self._stop_status()
            dash = glyph("dash")
            self.console.print(
                f"[{MUTED}]{dash} step {payload['step']}/{payload['max_steps']} "
                f"{dash}[/{MUTED}]"
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
            f"[{WARN}]![/{WARN}] [{MUTED}]model wrote {payload['count']} tool call(s) "
            f"as text; recovered[/{MUTED}]"
        )

    def _on_tool_call(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        label, detail = describe_tool_call(payload["name"], payload["arguments"])
        if payload["name"] == "task_complete":
            return  # rendered by _on_complete instead
        marker = glyph("bullet")
        self.console.print(
            f"[{MARKER}]{marker}[/{MARKER}] [{TOOL}]{label}[/{TOOL}]  "
            f"[{DETAIL}]{detail}[/{DETAIL}]"
        )
        self._start_status("running")

    def _on_tool_result(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        output = payload["output"]
        failed = output.startswith("Error:") or (
            output.startswith("exit_code: ") and not output.startswith("exit_code: 0")
        )
        body = collapse(output)
        if not body.strip():
            return
        style = "red1" if failed else OUTPUT
        bar = glyph("bar")
        for line in body.splitlines():
            self.console.print(f"[{MUTED}]{bar}[/{MUTED}] [{style}]{line}[/{style}]")

    def _on_completion_deferred(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        self.console.print(
            f"[{WARN}]![/{WARN}] [{MUTED}]claimed done in the same turn as the work; "
            f"asked it to check the results first[/{MUTED}]"
        )

    def _on_complete(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        summary = payload.get("summary") or "done"
        self.console.print(f"\n[{OK}]{glyph('tick')}[/{OK}] [{NOTE}]{summary}[/{NOTE}]")

    def _on_stopped(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        # Answering a question in prose is a perfectly good outcome -- the reply
        # has already been printed. Only a silent stop is worth flagging.
        if not payload.get("text", "").strip():
            self.console.print(
                f"\n[{WARN}]{glyph('circle')} stopped without doing anything[/{WARN}]"
            )

    def _on_context_trimmed(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        if not payload.get("dropped"):
            return
        self.console.print(
            f"[{WARN}]![/{WARN}] [{MUTED}]dropped {payload['dropped']} old messages to "
            f"stay within the context limit[/{MUTED}]"
        )

    def _on_budget_exhausted(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        self.console.print(
            f"\n[{FAIL}] {glyph('cross')} token budget spent "
            f"({payload['tokens']:,}) [/{FAIL}] "
            f"[{MUTED}]raise it with --max-tokens[/{MUTED}]"
        )

    def _on_max_iterations(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        self.console.print(
            f"\n[{FAIL}] {glyph('cross')} hit the {payload['step']}-step limit [/{FAIL}] "
            f"[{MUTED}]raise it with --max-iterations[/{MUTED}]"
        )

    def _on_error(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        self.console.print(
            f"\n[{FAIL}] {glyph('cross')} error [/{FAIL}] [red1]{payload['message']}[/red1]"
        )


def banner(
    console: Console,
    model: str,
    sandbox: str,
    mounts: list[tuple[str, str]],
    local: bool,
) -> None:
    lines = [
        Text.from_markup(f"[{BRAND}]cli-agent[/{BRAND}]  [{MUTED}]{model}[/{MUTED}]")
    ]

    if local:
        lines.append(
            Text.from_markup(f"[{FAIL}] running on your machine, unsandboxed [/{FAIL}]")
        )
    else:
        lines.append(Text.from_markup(f"[{MUTED}]sandbox {sandbox}[/{MUTED}]"))

    if mounts:
        for host, target in mounts:
            lines.append(
                Text.from_markup(
                    f"[{NOTE}]{target}[/{NOTE}] [{MUTED}]{glyph('arrow')}[/{MUTED}] "
                    f"[{MUTED}]{host}[/{MUTED}]"
                )
            )
        lines.append(Text.from_markup(f"[{MUTED}]files written there persist[/{MUTED}]"))
    elif not local:
        # They have lost files to this twice. Say it before the first turn.
        lines.append(
            Text.from_markup(
                f"[{WARN}]no folder mounted {glyph('dash')} files are discarded on "
                f"exit[/{WARN}]"
            )
        )
        lines.append(
            Text.from_markup(f"[{MUTED}]restart with --mount DIR to keep them[/{MUTED}]")
        )

    # Don't rely on rich's terminal detection for the border: we already know
    # whether this console can carry box-drawing characters.
    console.print(
        Panel(
            Group(*lines),
            border_style=BORDER,
            padding=(0, 2),
            box=rich_box.ASCII if ascii_only() else rich_box.HEAVY,
        )
    )
    dot = glyph("dot")
    console.print(
        f"[{MUTED}]/help for commands {dot} Ctrl+C interrupts a turn {dot} /exit to "
        f"quit[/{MUTED}]\n"
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
        bits.append(f"[red1]{m['tool_errors']} errors[/red1]")
    if m["recovered_tool_calls"]:
        bits.append(f"[{WARN}]{m['recovered_tool_calls']} recovered[/{WARN}]")
    separator = f"[{MUTED}] %s [/{MUTED}]" % glyph("dot")
    console.print(f"[{MUTED}]%s[/{MUTED}]\n" % separator.join(bits))
