"""Terminal rendering.

Turns the loop's event stream into something readable. The loop itself stays
UI-free -- it emits events and this decides how they look, so the benchmark
adapter can run the identical loop with no console attached at all.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from rich import box as rich_box
from rich.cells import cell_len
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
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


def humanize_error(message: str) -> tuple[str, str]:
    """Turn an API error into a headline and a next step.

    Raw provider errors arrive as a wall of JSON with the useful part buried in
    the middle. Returns (headline, hint); hint may be empty.
    """
    text = message or "unknown error"
    low = text.lower()

    if "rate limit" in low or "429" in low:
        used = re.search(r"Used (\d+)", text)
        limit = re.search(r"Limit (\d+)", text)
        retry = re.search(r"try again in ([0-9hms.]+)", text)
        if "per day" in low or "tpd" in low:
            headline = "daily token quota exhausted"
            detail = (
                f"used {int(used.group(1)):,} of {int(limit.group(1)):,} tokens today"
                if used and limit
                else "the free tier resets on a rolling 24h window"
            )
            return headline, f"{detail} — try a different model, or wait for the reset"
        headline = "rate limited"
        return headline, (
            f"retry in {retry.group(1)}" if retry else "too many requests just now"
        )

    if "api key" in low or "authentication" in low or "401" in low:
        return "API key rejected", "check GROQ_API_KEY in .env"
    if "connection" in low or "timed out" in low or "timeout" in low:
        return "could not reach the API", "network or provider issue; retried already"
    if "model" in low and ("not found" in low or "does not exist" in low):
        return "unknown model", "check --model"

    # Unrecognised: show it, but keep it to one readable line.
    flattened = " ".join(text.split())
    return (flattened[:200] + "…") if len(flattened) > 200 else flattened, ""


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
        headline, hint = humanize_error(payload["message"])
        self.console.print(f"\n[{FAIL}] {glyph('cross')} {headline} [/{FAIL}]")
        if hint:
            self.console.print(f"  [{MUTED}]{hint}[/{MUTED}]")


# The wordmark, and a plain fallback for consoles that cannot draw blocks.
_LOGO = r"""
 ██████╗ ██╗███████╗████████╗ ██████╗ ██████╗ ██████╗ ███████╗
 ██╔══██╗██║██╔════╝╚══██╔══╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
 ██║  ██║██║█████╗     ██║   ██║     ██║   ██║██║  ██║█████╗
 ██║  ██║██║██╔══╝     ██║   ██║     ██║   ██║██║  ██║██╔══╝
 ██████╔╝██║███████╗   ██║   ╚██████╗╚██████╔╝██████╔╝███████╗
 ╚═════╝ ╚═╝╚══════╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
""".strip("\n")

_LOGO_ASCII = r"""
  ___  _ ___ _____ ___ ___  ___  ___
 |   \| | __|_   _/ __/ _ \|   \| __|
 | |) | | _|  | || (_| (_) | |) | _|
 |___/|_|___| |_| \___\___/|___/|___|
""".strip("\n")

# Bright at the top, deepening downwards. Reads as lit-from-above rather than a
# flat block of red.
_LOGO_RAMP = ["#ff8a80", "#ff5252", "#ff1744", "#e51230", "#c20e26", "#96091d"]


def logo_lines() -> list[Text]:
    """The wordmark as individual lines, so it can sit in a column."""
    art = _LOGO_ASCII if ascii_only() else _LOGO
    return [
        Text(line, style=_LOGO_RAMP[min(i, len(_LOGO_RAMP) - 1)])
        for i, line in enumerate(art.splitlines())
    ]


def logo() -> Text:
    """The wordmark, coloured as a vertical gradient."""
    text = Text()
    for line in logo_lines():
        text.append_text(line)
        text.append("\n")
    return text


# cell_len, not len: rich measures some of these box/block glyphs as wider than
# one column, and sizing the column with len() truncates the wordmark.
LOGO_WIDTH = max(cell_len(line) for line in _LOGO.splitlines())


def recent_activity(mounts: list[tuple[str, str]], limit: int = 3) -> list[str]:
    """What the agent last left behind in the mounted folder.

    Genuinely useful rather than decorative: it answers "did my files actually
    land?" without leaving the prompt -- the question that cost real work here
    more than once.
    """
    if not mounts:
        return []
    from pathlib import Path

    try:
        root = Path(mounts[0][0])
        files = [p for p in root.iterdir() if p.is_file()]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [f"{p.name}  ({p.stat().st_size:,}b)" for p in files[:limit]]


def context_percent(used: int, budget: int) -> int:
    """How much of the context budget is still free, as a percentage."""
    if budget <= 0:
        return 100
    return max(0, min(100, round((1 - used / budget) * 100)))


def status_bar(
    location: str,
    sandbox: str,
    model: str,
    context_left: int,
    width: int = 80,
) -> str:
    """The pinned line under the input: shortcuts on the left, state on the
    right. Raw ANSI because prompt_toolkit renders this, not rich."""
    esc = "\x1b["
    dim, red, orange, green, reset = (
        f"{esc}38;5;244m",
        f"{esc}38;5;203m",
        f"{esc}38;5;208m",
        f"{esc}38;5;114m",
        f"{esc}0m",
    )
    # Green while there is room, orange as it tightens, red when trimming is
    # imminent -- the number matters most exactly when it is small.
    ctx_colour = green if context_left > 50 else orange if context_left > 20 else red

    left = f"{dim}/help for commands{reset}"
    right_plain = f"{location}  {_strip_ansi(sandbox)}  {model} ({context_left}%)"
    right = (
        f"{dim}{location}{reset}  {sandbox}  "
        f"{dim}{model}{reset} {ctx_colour}({context_left}%){reset}"
    )

    gap = max(2, width - len("/help for commands") - len(right_plain) - 2)
    return f" {left}{' ' * gap}{right} "


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def sandbox_label(container: str | None, mounts: list[tuple[str, str]], local: bool) -> str:
    """Isolation state, in the colour it deserves."""
    esc = "\x1b["
    red, orange, green, reset = (
        f"{esc}38;5;203m",
        f"{esc}38;5;208m",
        f"{esc}38;5;114m",
        f"{esc}0m",
    )
    if local:
        return f"{red}no sandbox{reset}"
    if mounts:
        return f"{green}sandboxed{reset} {orange}+ mounted{reset}"
    return f"{green}sandboxed{reset}"


# Kept short so they fit the guide column without being cut mid-word.
TIPS = [
    "Describe a task, it runs until done.",
    "Be specific; it can install & test.",
    f"[{TOOL}]/help[/{TOOL}] for commands, [{TOOL}]/exit[/{TOOL}] to quit.",
]


def banner(
    console: Console,
    model: str,
    sandbox: str,
    mounts: list[tuple[str, str]],
    local: bool,
) -> None:
    left = _identity_block(model, sandbox, mounts, local)
    right = _guide_block(mounts, local)

    # Two columns only when the guide column stays readable; stacked otherwise.
    # A squeezed two-column layout is worse than an honest single one.
    if _guide_width(console.width) >= 34:
        body: Any = _side_by_side(left, right, console.width)
    else:
        body = Group(*left, Text(""), *right)

    console.print()
    console.print(
        Panel(
            body,
            title=f"[{BRAND}]dietcode[/{BRAND}] [{MUTED}]v{__version__}[/{MUTED}]",
            title_align="left",
            border_style=BORDER,
            padding=(1, 2),
            box=rich_box.ASCII if ascii_only() else rich_box.ROUNDED,
        )
    )

    # The one-line notice under the panel, where Claude Code puts its news.
    if local:
        # Working in your own files is the normal case now, so this reports the
        # protection rather than raising an alarm. The genuinely dangerous state
        # is --yes, which the CLI shouts about separately.
        console.print(
            f" [{NOTE}]{glyph('arrow')}[/{NOTE}] [{MUTED}]working in your files "
            f"{glyph('dot')} asks before anything that writes, deletes, or "
            f"leaves this folder[/{MUTED}]\n",
            no_wrap=True,
            overflow="ellipsis",
        )
    elif not mounts:
        console.print(
            f" [{WARN}]{glyph('arrow')}[/{WARN}] [{MUTED}]nothing mounted "
            f"{glyph('dot')} files vanish on exit "
            f"{glyph('dot')} use --mount DIR to keep them[/{MUTED}]\n",
            no_wrap=True,
            overflow="ellipsis",
        )
    else:
        console.print(
            f" [{NOTE}]{glyph('arrow')}[/{NOTE}] [{MUTED}]files land in "
            f"{_shorten(mounts[0][0], max(20, console.width - 22))}[/{MUTED}]\n",
            no_wrap=True,
            overflow="ellipsis",
        )


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else "..." + text[-(width - 3) :]


def _identity_block(
    model: str, sandbox: str, mounts: list[tuple[str, str]], local: bool
) -> list[Text]:
    """Left column: the wordmark, then what this session actually is."""
    lines: list[Text] = list(logo_lines())
    lines.append(Text(""))

    lines.append(
        Text.from_markup(
            f"[{NOTE}]{model}[/{NOTE}] [{MUTED}]{glyph('dot')}[/{MUTED}] "
            + (
                f"[{DETAIL}]asks first[/{DETAIL}]"
                if local
                else f"[{DETAIL}]sandboxed[/{DETAIL}]"
            )
        )
    )
    # `sandbox` is the container name when there is one, the working directory
    # otherwise -- either way it is the answer to "where is this happening".
    lines.append(
        Text(
            _shorten(sandbox, LOGO_WIDTH - 4) if local else f"container {sandbox[:26]}",
            style=MUTED,
        )
    )
    for host, target in mounts:
        lines.append(
            Text.from_markup(
                f"[{MUTED}]{target} {glyph('arrow')} {_shorten(host, LOGO_WIDTH - 14)}[/{MUTED}]"
            )
        )
    return lines


def _guide_block(mounts: list[tuple[str, str]], local: bool) -> list[Text]:
    """Right column: how to start, and what is already here."""
    lines = [Text.from_markup(f"[{TOOL}]Tips for getting started[/{TOOL}]")]
    for tip in TIPS:
        lines.append(Text.from_markup(f"[{DETAIL}]{tip}[/{DETAIL}]"))

    lines.append(Text(""))
    lines.append(Text.from_markup(f"[{TOOL}]Recent activity[/{TOOL}]"))
    recent = recent_activity(mounts)
    if recent:
        for entry in recent:
            lines.append(Text(entry, style=MUTED))
    else:
        lines.append(Text("No recent activity", style=MUTED))
    return lines


# Panel borders (2) plus its horizontal padding (2 each side).
_PANEL_CHROME = 6
# Grid padding either side of the divider, plus the divider itself.
_COLUMN_GAP = 5


def _guide_width(console_width: int) -> int:
    return console_width - _PANEL_CHROME - LOGO_WIDTH - _COLUMN_GAP


def _side_by_side(left: list[Text], right: list[Text], console_width: int) -> Table:
    """Align two ragged columns with a rule between them.

    Every column is given an explicit width. Left to itself, rich shrinks the
    fixed wordmark column to make room for the flexible one, which truncates
    the logo mid-glyph.
    """
    height = max(len(left), len(right))
    left = left + [Text("")] * (height - len(left))
    right = right + [Text("")] * (height - len(right))

    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=LOGO_WIDTH, no_wrap=True, overflow="crop")
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(width=_guide_width(console_width), overflow="ellipsis", no_wrap=True)
    divider = Text(glyph("bar"), style=BORDER)
    for i in range(height):
        grid.add_row(left[i], divider, right[i])
    return grid


def make_approver(console: Console, renderer: Renderer | None = None):
    """Build the approval prompt for host mode.

    Lives here rather than in permissions.py so the gate stays free of any
    rendering, the same way the loop is.
    """
    from .permissions import Decision, Request, Risk

    risk_style = {
        Risk.READ_ONLY: DETAIL,
        Risk.MODIFIES: WARN,
        Risk.DANGEROUS: FAIL,
    }

    def ask(request: Request) -> Decision:
        if renderer is not None:
            renderer.close()  # stop the spinner before taking over the line

        verb = {"run": "run", "write": "write to", "read": "read"}[request.action]
        style = risk_style[request.risk]
        console.print()
        console.print(
            f"[{style}] permission [/{style}] [{TOOL}]{verb}[/{TOOL}] "
            f"[{DETAIL}]{request.detail}[/{DETAIL}]"
        )
        if request.outside_root:
            console.print(
                f"  [{FAIL}] outside {request.root} [/{FAIL}]",
            )
        if request.risk is Risk.DANGEROUS:
            console.print(f"  [{WARN}]this is a destructive operation[/{WARN}]")

        key = request.remember_key.split(":", 1)[1]
        console.print(
            f"  [{MUTED}][[/{MUTED}][{OK}]y[/{OK}][{MUTED}]] yes  "
            f"[[/{MUTED}][{TOOL}]a[/{TOOL}][{MUTED}]] always allow "
            f"{key}  [[/{MUTED}][{WARN}]n[/{WARN}][{MUTED}]] no[/{MUTED}]"
        )

        try:
            answer = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print(f"  [{WARN}]denied[/{WARN}]")
            return Decision.NO

        if answer in ("a", "always"):
            return Decision.ALWAYS
        if answer in ("y", "yes"):
            return Decision.ONCE
        return Decision.NO

    return ask


def input_rule(console: Console) -> None:
    """The rule above the prompt. prompt_toolkit's bottom toolbar closes the
    frame underneath, giving the input a top and bottom edge.

    Rules rather than a real box: a four-sided box needs a full-screen
    prompt_toolkit application, which takes over the terminal and destroys
    scrollback. Not worth losing scrollback for two vertical lines.
    """
    line = "-" if ascii_only() else "─"
    console.print(f"[{BORDER}]{line * max(4, console.width - 1)}[/{BORDER}]")


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
