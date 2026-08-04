"""The `dietcode login` / `logout` / `auth` subcommands.

Kept apart from cli.py because these are the only commands that run without a
sandbox, a model, or a task -- mixing them into the agent entrypoint made both
harder to read.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from .auth import (
    PROVIDERS,
    AuthError,
    clean_key,
    credential_status,
    default_provider,
    delete_key,
    get_provider,
    mask,
    resolve_key,
    set_default_provider,
    store_key,
)
from .ui import BRAND, DETAIL, FAIL, MUTED, NOTE, OK, TOOL, WARN, glyph


def _prompt_secret(console: Console, label: str) -> str:
    """Read a key without echoing it."""
    import getpass

    if not sys.stdin.isatty():
        # Piped input: read a line so `echo $KEY | dietcode login` works.
        return sys.stdin.readline().strip()
    try:
        return getpass.getpass(f"{label}: ")
    except (EOFError, KeyboardInterrupt):
        console.print()
        return ""


def login(console: Console, provider: str | None, api_key: str | None) -> int:
    """Save an API key for a provider."""
    if provider is None:
        console.print(f"[{TOOL}]Which provider?[/{TOOL}]")
        for spec in PROVIDERS.values():
            console.print(
                f"  [{NOTE}]{spec.name:<8}[/{NOTE}] [{MUTED}]{spec.label} "
                f"{glyph('dot')} keys at {spec.signup_url}[/{MUTED}]"
            )
        console.print()
        try:
            provider = input(f"provider [{list(PROVIDERS)[0]}]: ").strip() or list(PROVIDERS)[0]
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 130

    try:
        spec = get_provider(provider)
    except AuthError as exc:
        console.print(f"[{FAIL}] {exc} [/{FAIL}]")
        return 2

    if not api_key:
        console.print(
            f"[{MUTED}]Get a key at {spec.signup_url} "
            f"{glyph('dot')} input is hidden[/{MUTED}]"
        )
        api_key = _prompt_secret(console, f"{spec.label} API key")

    api_key = clean_key(api_key or "")
    if not api_key:
        console.print(f"[{WARN}]no key entered, nothing saved[/{WARN}]")
        return 1

    # A wrong-provider paste is a common slip and produces a confusing 401
    # later; say so now, but do not refuse -- key formats change.
    if spec.key_hint and not api_key.startswith(spec.key_hint):
        console.print(
            f"[{WARN}]note: {spec.label} keys usually start with "
            f"{spec.key_hint!r}[/{WARN}]"
        )

    try:
        where = store_key(spec.name, api_key)
    except AuthError as exc:
        console.print(f"[{FAIL}] {exc} [/{FAIL}]")
        return 2

    set_default_provider(spec.name)
    console.print(
        f"[{OK}]{glyph('tick')}[/{OK}] saved {spec.label} key "
        f"[{MUTED}]({mask(api_key)}) to {where}[/{MUTED}]"
    )
    console.print(f"[{MUTED}]default model: {spec.default_model}[/{MUTED}]")
    return 0


def logout(console: Console, provider: str | None) -> int:
    """Forget stored keys."""
    targets = [provider] if provider else list(PROVIDERS)
    removed = []
    for name in targets:
        try:
            get_provider(name)
        except AuthError as exc:
            console.print(f"[{FAIL}] {exc} [/{FAIL}]")
            return 2
        if delete_key(name):
            removed.append(name)

    if not removed:
        console.print(f"[{MUTED}]nothing stored to remove[/{MUTED}]")
        return 0

    console.print(f"[{OK}]{glyph('tick')}[/{OK}] removed: {', '.join(removed)}")
    # An env var outlives logout, and silently keeps you logged in.
    for name in removed:
        key, source = resolve_key(name)
        if key and source.startswith("$"):
            console.print(
                f"[{WARN}]{name} is still authenticated via {source}[/{WARN}]"
            )
    return 0


def doctor(console: Console) -> int:
    """Check everything a fresh install needs, and say how to fix what is missing.

    Worth its weight: almost every "it doesn't work" on someone else's laptop is
    one of these four things, and none of them produce an obvious error on their
    own.
    """
    import shutil
    import subprocess

    from . import __version__

    ok = True

    def report(good: bool, label: str, detail: str, fix: str = "") -> None:
        nonlocal ok
        mark = f"[{OK}]{glyph('tick')}[/{OK}]" if good else f"[{FAIL}] {glyph('cross')} [/{FAIL}]"
        console.print(f"{mark} [{TOOL}]{label:<12}[/{TOOL}] [{MUTED}]{detail}[/{MUTED}]")
        if not good:
            ok = False
            if fix:
                console.print(f"     [{NOTE}]{fix}[/{NOTE}]")

    console.print(f"[{BRAND}]dietcode {__version__}[/{BRAND}]\n")

    version = sys.version_info
    report(
        version >= (3, 11),
        "python",
        f"{version.major}.{version.minor}.{version.micro}",
        "dietcode needs Python 3.11 or newer",
    )

    # The command being on PATH is the single most common first-run problem,
    # and the error it produces ("not recognized") never mentions Python.
    on_path = shutil.which("dietcode")
    report(
        on_path is not None,
        "on PATH",
        on_path or "dietcode is not on PATH",
        "install with `pipx install dietcode`, which handles PATH for you",
    )

    docker = shutil.which("docker")
    if docker is None:
        report(
            False,
            "docker",
            "not installed",
            "optional: without it use `dietcode --here` to work in the current "
            "directory instead of a container",
        )
    else:
        try:
            proc = subprocess.run(
                [docker, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                timeout=20,
            )
            running = proc.returncode == 0
            detail = (
                proc.stdout.decode(errors="replace").strip() if running else "installed, not running"
            )
        except (subprocess.SubprocessError, OSError):
            running, detail = False, "installed, not responding"
        report(
            running,
            "docker",
            detail,
            "start Docker Desktop, or use `dietcode --here` to skip the sandbox",
        )

    configured = [spec.name for spec, masked, _ in credential_status() if masked]
    report(
        bool(configured),
        "credentials",
        ", ".join(configured) if configured else "none",
        "run `dietcode login`",
    )

    console.print()
    if ok:
        console.print(f"[{OK}]{glyph('tick')} ready[/{OK}] [{MUTED}]try: dietcode --here[/{MUTED}]")
        return 0
    console.print(f"[{WARN}]fix the items above, then run `dietcode doctor` again[/{WARN}]")
    return 1


def auth_status(console: Console) -> int:
    """Show which providers are usable and where each key comes from."""
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("provider", style=NOTE)
    table.add_column("key", style=MUTED)
    table.add_column("source", style=MUTED)
    table.add_column("default model", style=DETAIL)

    current = default_provider()
    any_key = False
    for spec, masked, source in credential_status():
        any_key = any_key or masked is not None
        marker = f" [{BRAND}]{glyph('bullet')}[/{BRAND}]" if spec.name == current else ""
        table.add_row(
            f"{spec.name}{marker}",
            masked or f"[{MUTED}]—[/{MUTED}]",
            source,
            spec.default_model,
        )

    console.print(table)
    if not any_key:
        console.print(
            f"\n[{WARN}]No credentials.[/{WARN}] "
            f"[{MUTED}]Run [/{MUTED}][{TOOL}]dietcode login[/{TOOL}]"
            f"[{MUTED}] to save one.[/{MUTED}]"
        )
        return 1
    console.print(
        f"\n[{MUTED}]{glyph('bullet')} marks the provider used by default "
        f"{glyph('dot')} override with --provider[/{MUTED}]"
    )
    return 0
