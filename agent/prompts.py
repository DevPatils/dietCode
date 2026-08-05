"""Interactive pickers and prompts.

Typing a provider name at a `provider [groq]:` prompt works, but it makes the
user do the remembering. These are arrow-key choosers instead, rendered inline
so they scroll away with everything else rather than taking over the screen.

Everything degrades: no prompt_toolkit falls back to plain typed input, and a
run with no terminal cancels rather than hanging or guessing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from .ui import ascii_only

# A provider can list sixty-odd models. Showing them all would push the prompt
# off the top of the screen, so the list scrolls inside a fixed window.
VISIBLE_ROWS = 8


@dataclass
class Choice:
    value: str
    label: str
    hint: str = ""


def interactive() -> bool:
    """Whether there is a human at a terminal to answer."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _start_index(choices: list[Choice], default: int, selected: str | None) -> int:
    """Open with the cursor on what is already in use, not on row one."""
    if selected is not None:
        for i, choice in enumerate(choices):
            if choice.value == selected:
                return i
    return min(default, len(choices) - 1)


def choose(
    console: Console,
    title: str,
    choices: list[Choice],
    default: int = 0,
    selected: str | None = None,
) -> str | None:
    """Pick one option. Returns the value, or None if cancelled.

    Arrow keys and j/k move, typing filters, Enter confirms, Esc cancels.
    """
    if not choices:
        return None
    start = _start_index(choices, default, selected)
    if not interactive():
        # No terminal means nobody can answer. Guessing the first option would
        # silently change a setting in a scripted run, so treat it as cancelled
        # and let the caller say what flag to pass instead.
        return None

    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style
    except ImportError:
        return _typed_fallback(console, title, choices, start)

    index = start
    query = ""
    visible = choices
    width = min(max((len(c.label) for c in choices), default=8) + 2, 40)
    scrollable = len(choices) > VISIBLE_ROWS

    # cp1252 consoles cannot encode the pointer or the arrows; printing one
    # raises mid-render and takes the picker down with it.
    plain = ascii_only()
    cursor = "  > " if plain else "  ❯ "
    arrows = "up/down" if plain else "↑↓"
    dot = "-" if plain else "·"

    console.print(f"[bold]{title}[/bold]")

    def refilter() -> None:
        """Narrow the list to what matches, keeping the cursor on-screen."""
        nonlocal visible, index
        previous = visible[index].value if visible and index < len(visible) else None
        visible = [c for c in choices if query.lower() in c.label.lower()] if query else choices
        index = 0
        if previous is not None:
            for i, choice in enumerate(visible):
                if choice.value == previous:
                    index = i
                    break

    def window() -> tuple[int, int]:
        """The slice of the list to draw, scrolled to keep the cursor inside."""
        if len(visible) <= VISIBLE_ROWS:
            return 0, len(visible)
        top = min(max(0, index - VISIBLE_ROWS // 2), len(visible) - VISIBLE_ROWS)
        return top, top + VISIBLE_ROWS

    def render() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if not visible:
            out.append(("class:hint", f"  no match for {query!r}\n"))
        top, bottom = window()
        if top > 0:
            out.append(("class:dim", f"    {top} more above\n"))
        for i in range(top, bottom):
            choice = visible[i]
            is_cursor = i == index
            out.append(
                ("class:marker" if is_cursor else "class:dim", cursor if is_cursor else "    ")
            )
            out.append(
                (
                    "class:selected" if is_cursor else "class:option",
                    f"{choice.label:<{width}}",
                )
            )
            out.append(("class:hint", f" {choice.hint}\n"))
        if bottom < len(visible):
            out.append(("class:dim", f"    {len(visible) - bottom} more below\n"))

        footer = f"  {arrows} to move {dot} enter to choose {dot} esc to cancel"
        if scrollable:
            footer = (
                f"  {arrows} to move {dot} type to filter {dot} "
                f"enter to choose {dot} esc to cancel"
            )
        if query:
            out.append(("class:filter", f"\n  filter: {query}"))
        out.append(("class:dim", f"\n{footer}"))
        return out

    bindings = KeyBindings()

    def move(step: int) -> None:
        nonlocal index
        if visible:
            index = (index + step) % len(visible)

    @bindings.add("up")
    def _up(_event: Any) -> None:
        move(-1)

    @bindings.add("down")
    def _down(_event: Any) -> None:
        move(1)

    @bindings.add("pageup")
    def _page_up(_event: Any) -> None:
        move(-VISIBLE_ROWS)

    @bindings.add("pagedown")
    def _page_down(_event: Any) -> None:
        move(VISIBLE_ROWS)

    @bindings.add("enter")
    def _accept(event: Any) -> None:
        event.app.exit(result=visible[index].value if visible else None)

    @bindings.add("escape", eager=True)
    @bindings.add("c-c")
    @bindings.add("c-d")
    def _cancel(event: Any) -> None:
        event.app.exit(result=None)

    if scrollable:
        # Long lists get type-to-filter; j/k and digits would be swallowed as
        # filter text, which is the right trade when there are sixty options.
        @bindings.add("backspace")
        def _backspace(_event: Any) -> None:
            nonlocal query
            query = query[:-1]
            refilter()

        @bindings.add("<any>")
        def _type(event: Any) -> None:
            nonlocal query
            text = event.data
            if text and text.isprintable():
                query += text
                refilter()
    else:
        @bindings.add("k")
        def _up_k(_event: Any) -> None:
            move(-1)

        @bindings.add("j")
        def _down_j(_event: Any) -> None:
            move(1)

        # Number keys, because reaching for the arrows is slower when you
        # already know which one you want.
        for n in range(1, min(len(choices), 9) + 1):
            @bindings.add(str(n))
            def _pick(event: Any, n: int = n) -> None:
                event.app.exit(result=choices[n - 1].value)

    app: Application[str | None] = Application(
        layout=Layout(
            HSplit([Window(FormattedTextControl(render), dont_extend_height=True)])
        ),
        key_bindings=bindings,
        style=Style.from_dict(
            {
                "marker": "#ff5252 bold",
                "selected": "#ff8a80 bold",
                "option": "",
                "hint": "#8a8a8a",
                "filter": "#ff8a80",
                "dim": "#6e6669",
            }
        ),
        # Inline, not full screen: the choice stays in the scrollback with the
        # rest of the session instead of blanking the terminal.
        full_screen=False,
        erase_when_done=False,
    )
    try:
        return app.run()
    except Exception:  # noqa: BLE001 - odd terminals; typing still works
        return _typed_fallback(console, title, choices, start)


def _typed_fallback(
    console: Console, title: str, choices: list[Choice], default: int
) -> str | None:
    console.print(f"[bold]{title}[/bold]")
    for i, choice in enumerate(choices, 1):
        console.print(f"  [dim]{i}.[/dim] {choice.label}  [dim]{choice.hint}[/dim]")
    try:
        answer = input(f"choice [{choices[default].label}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None
    if not answer:
        return choices[default].value
    if answer.isdigit() and 1 <= int(answer) <= len(choices):
        return choices[int(answer) - 1].value
    for choice in choices:
        if answer.lower() in (choice.value.lower(), choice.label.lower()):
            return choice.value
    console.print(f"[red1]not one of the options: {answer}[/red1]")
    return None


def ask_secret(label: str) -> str:
    """Read a secret without echoing it, or from a pipe when there is no tty."""
    if not sys.stdin.isatty():
        return sys.stdin.readline().strip()
    import getpass

    try:
        return getpass.getpass(f"{label}: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def confirm(console: Console, question: str, default: bool = True) -> bool:
    """A yes/no that shows what you typed.

    The obvious implementation -- input() -- gets overwritten when a spinner is
    still painting, so the user types blind. prompt_toolkit owns the line and
    redraws around its own output.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    if not interactive():
        return default
    try:
        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.formatted_text import ANSI

        answer = pt_prompt(ANSI(f"\x1b[1m{question}\x1b[0m \x1b[38;5;244m{suffix}\x1b[0m "))
    except ImportError:
        try:
            answer = input(f"{question} {suffix} ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    answer = answer.strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")
