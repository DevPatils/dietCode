"""Entrypoint for the `dietcode` command.

    dietcode                      interactive session
    dietcode "do the thing"       one shot, then exit
    dietcode login                save an API key
    dietcode auth                 show which providers are usable

Every mode runs the same agent_loop against the same sandbox; the interactive
path just keeps the container and the conversation alive between turns.
"""

from __future__ import annotations

import argparse
import atexit
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from . import __version__
from .auth import PROVIDERS, AuthError, default_provider, get_provider, resolve_key
from .commands import auth_status, doctor, login, logout
from .loop import (
    DEFAULT_CONTEXT_BUDGET,
    DEFAULT_MAX_ITERATIONS,
    SYSTEM_PROMPT,
    agent_loop,
    ensure_project_context,
    load_project_context,
    make_client,
    with_project_context,
)
from .memory import REMEMBER_TOOL, make_remember_handler, with_memory
from .permissions import MODE_HELP, Mode, PermissionGate, Policy, deny_all
from .prompts import Choice, choose, confirm
from .repl import Session
from .sandbox import (
    DEFAULT_CPUS,
    DEFAULT_IMAGE,
    DEFAULT_MEMORY,
    DEFAULT_PIDS_LIMIT,
    DockerExecutor,
    LocalExecutor,
    SandboxError,
)
from .sessions import (
    SessionStore,
    find_session,
    fork,
    latest_session,
    list_sessions,
    load_messages,
)
from .snapshots import SnapshotStore, Snapshotting
from .subagent import SPAWN_TOOL, make_spawn_handler
from .tools import tools_for
from .ui import FAIL, Renderer, make_approver, turn_footer, use_utf8_stdout

SUBCOMMANDS = {"login", "logout", "auth", "doctor"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dietcode",
        description="A coding agent that writes and runs code in a sandbox.",
        epilog="Run `dietcode login` first, then `dietcode` for an interactive session.",
    )
    parser.add_argument("task", nargs="?", help="what to do; omit for interactive mode")
    parser.add_argument("--version", action="version", version=f"dietcode {__version__}")

    model = parser.add_argument_group("model")
    model.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        help="which API to use (default: your saved login)",
    )
    model.add_argument("--model", help="override the provider's default model")
    model.add_argument("--base-url", help="use an OpenAI-compatible endpoint directly")
    model.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    model.add_argument(
        "--subagents",
        action="store_true",
        help="let the agent delegate self-contained work to sub-agents",
    )
    model.add_argument(
        "--verify",
        metavar="CMD",
        help="a command that must exit 0 before the agent is allowed to finish, "
        'e.g. --verify "python -m pytest -q"',
    )
    model.add_argument(
        "--no-memory",
        dest="memory",
        action="store_false",
        help="do not load or write the agent's notes for this project",
    )
    model.add_argument(
        "--no-context",
        dest="context",
        action="store_false",
        help="ignore DIETCODE.md / AGENTS.md / CLAUDE.md in the working directory",
    )
    model.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="stop once this many tokens have been spent (per task)",
    )
    model.add_argument(
        "--context-budget",
        type=int,
        default=DEFAULT_CONTEXT_BUDGET,
        metavar="N",
        help=f"trim the oldest turns above this prompt size (default {DEFAULT_CONTEXT_BUDGET})",
    )

    sandbox = parser.add_argument_group("where it runs")
    # Default is the current directory. Docker is opt-in, because most people
    # want an agent that works on the project they are standing in, and the
    # permission gate is what keeps that safe.
    sandbox.add_argument(
        "--sandbox",
        action="store_true",
        help="run inside a Docker container instead of the current directory",
    )
    sandbox.add_argument(
        "--mount",
        action="append",
        metavar="HOSTDIR[:TARGET]",
        help="with --sandbox, bind-mount a host directory into the container "
        "(default target /workspace). Repeatable. Implies --sandbox.",
    )
    sandbox.add_argument(
        "--here",
        "--local",
        dest="here",
        action="store_true",
        help=argparse.SUPPRESS,  # now the default; kept so old commands still work
    )
    sandbox.add_argument("--workdir", default=".", help="directory to work in")
    sandbox.add_argument(
        "--mode",
        choices=[m.value for m in Mode],
        default=Mode.MANUAL.value,
        help="how much to do before asking: "
        + " | ".join(f"{m.value} ({MODE_HELP[m]})" for m in Mode),
    )
    sandbox.add_argument(
        "--yes",
        action="store_true",
        help="approve every action without asking (same as --mode auto)",
    )
    sandbox.add_argument("--image", default=DEFAULT_IMAGE, help="sandbox container image")
    sandbox.add_argument(
        "--container", help="attach to an existing container instead of creating one"
    )
    sandbox.add_argument("--memory", default=DEFAULT_MEMORY, help="container memory cap")
    sandbox.add_argument("--cpus", default=DEFAULT_CPUS, help="container CPU cap")
    sandbox.add_argument(
        "--pids-limit", type=int, default=DEFAULT_PIDS_LIMIT, help="container process cap"
    )
    sandbox.add_argument(
        "--no-network",
        action="store_true",
        help="cut the sandbox off from the network entirely",
    )
    sandbox.add_argument(
        "--cleanup",
        action="store_true",
        help="remove every leftover agent container and exit",
    )

    session = parser.add_argument_group("sessions")
    session.add_argument(
        "--resume",
        nargs="?",
        const="",
        metavar="ID",
        help="carry on a saved session; omit the id to pick from a list",
    )
    session.add_argument(
        "--continue",
        dest="continue_last",
        action="store_true",
        help="carry on the most recent session in this directory",
    )
    session.add_argument(
        "--fork",
        metavar="ID",
        help="branch a saved session into a new one and continue there",
    )
    session.add_argument(
        "--no-snapshots",
        dest="snapshots",
        action="store_false",
        help="do not keep a copy of each file before the agent changes it",
    )
    session.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="do not write a transcript for this session",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--steps", action="store_true", help="show step separators")
    output.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="wait for each reply instead of showing it as it is generated",
    )
    output.add_argument("--quiet", action="store_true", help="only print the final result")
    output.add_argument("--json", action="store_true", help="print metrics as JSON")
    return parser


def build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dietcode", add_help=True)
    subs = parser.add_subparsers(dest="command", required=True)

    p_login = subs.add_parser("login", help="save an API key for a provider")
    p_login.add_argument("provider", nargs="?", choices=sorted(PROVIDERS))
    p_login.add_argument("--key", help="the API key (otherwise prompted, hidden)")

    p_logout = subs.add_parser("logout", help="forget stored keys")
    p_logout.add_argument("provider", nargs="?", choices=sorted(PROVIDERS))

    subs.add_parser("auth", help="show which providers are usable")
    subs.add_parser("doctor", help="check this machine is set up correctly")
    return parser


def resolve_model_config(args: argparse.Namespace) -> tuple[str, str, str]:
    """(api_key, base_url, model), or raise AuthError explaining what is missing."""
    provider_name = args.provider or default_provider()
    spec = get_provider(provider_name)

    api_key, source = resolve_key(provider_name)
    if not api_key:
        raise AuthError(
            f"no credentials for {spec.label}.\n"
            f"Run `dietcode login {spec.name}`, or set ${spec.env_var}."
        )
    del source
    return api_key, args.base_url or spec.base_url, args.model or spec.default_model


def wants_sandbox(args: argparse.Namespace) -> bool:
    """Docker is opt-in. --mount only means anything inside a container, so it
    implies --sandbox rather than being silently ignored."""
    return bool(args.sandbox or args.container or args.mount or args.no_network)


def make_executor(
    args: argparse.Namespace,
    console: Console,
    renderer: Renderer | None = None,
    snapshots: SnapshotStore | None = None,
) -> tuple[Any, list]:
    if not wants_sandbox(args):
        root = Path(args.workdir).resolve()
        inner: Any = LocalExecutor(root)
        # Inside the gate, not outside it: the gate asks first, and only an
        # approved write ever reaches the snapshot. Wrapping the other way
        # round would prompt for a read every time a file was about to change.
        if snapshots is not None:
            inner = Snapshotting(inner, snapshots)

        # --yes is the older spelling of --mode auto; keep it working.
        mode = Mode.AUTO if args.yes else Mode(args.mode)

        if mode is Mode.AUTO:
            console.print(
                f"[{FAIL}] auto: every command runs without asking [/{FAIL}] "
                f"[orange3]{root}[/orange3]"
            )
            policy = Policy.for_mode(mode)
            approver = deny_all  # unreachable while yes_to_everything is set
        elif sys.stdin.isatty():
            policy = Policy.for_mode(mode)
            if mode is not Mode.MANUAL:
                console.print(f"[dim]{mode.value}: {MODE_HELP[mode]}[/dim]")
            approver = make_approver(console, renderer)
        else:
            # Nothing can be asked, so nothing destructive may happen. Silently
            # approving here is how an automated run rewrites someone's files.
            console.print(
                "[orange3]not a terminal: actions that need approval will be "
                "denied. Use --yes to override.[/orange3]"
            )
            policy = Policy.for_mode(mode)
            approver = deny_all

        return PermissionGate(inner, root=root, approver=approver, policy=policy), []

    mounts = [DockerExecutor.parse_mount(m) for m in (args.mount or [])]
    if mounts and args.container:
        console.print(
            "[yellow]--mount is ignored when attaching to an existing container[/yellow]"
        )
        mounts = []
    executor = DockerExecutor(
        image=args.image,
        container=args.container,
        mounts=mounts,
        memory=args.memory,
        cpus=args.cpus,
        pids_limit=args.pids_limit,
        network="none" if args.no_network else None,
    )
    if snapshots is not None:
        return Snapshotting(executor, snapshots), mounts
    return executor, mounts


def resolve_session(
    args: argparse.Namespace, console: Console, model: str
) -> tuple[Any, list[dict[str, Any]] | None]:
    """Work out which transcript this session writes to, and what it starts with.

    Returns (store, history). `(None, None)` means the user asked for a session
    that could not be found -- caller should give up rather than silently open a
    new one, because "resume" that quietly starts fresh loses work.
    """
    project = str(Path(args.workdir).resolve())
    target: Any = None

    if args.fork:
        source = find_session(project, args.fork)
        if source is None:
            console.print(f"[red]no session matching {args.fork!r} in this directory[/red]")
            return None, None
        forked = fork(source)
        console.print(f"[dim]forked {source.session_id} -> {forked}[/dim]")
        target = find_session(project, forked)
    elif args.continue_last:
        target = latest_session(project)
        if target is None:
            console.print("[dim]no earlier session here; starting a new one[/dim]")
    elif args.resume is not None:
        if args.resume:
            target = find_session(project, args.resume)
            if target is None:
                console.print(f"[red]no session matching {args.resume!r} in this directory[/red]")
                return None, None
        else:
            target = pick_session(console, project)
            if target is None:
                return None, None

    if not args.save:
        # Still resumable: reading a transcript and appending to it are
        # separate decisions.
        history = load_messages(target.path) if target else None
        return None, history

    if target is not None:
        store = SessionStore(
            project=project,
            session_id=target.session_id,
            model=model,
            provider=args.provider or default_provider(),
        )
        history = load_messages(target.path)
        # Already on disk; without this the whole conversation is appended a
        # second time on the next turn.
        store._written = len(history)
        store._wrote_header = True
        console.print(
            f"[dim]resumed {target.session_id} {len(history)} messages[/dim]"
        )
        return store, history

    return (
        SessionStore(
            project=project,
            model=model,
            provider=args.provider or default_provider(),
        ),
        None,
    )


def pick_session(console: Console, project: str) -> Any:
    """`--resume` with no id: choose from what is actually there."""
    rows = list_sessions(project)
    if not rows:
        console.print("[dim]no saved sessions in this directory[/dim]")
        return None
    chosen = choose(
        console,
        "Resume which session?",
        [
            Choice(
                meta.session_id,
                meta.session_id,
                f"{meta.turns} turn(s)  {meta.label}",
            )
            for meta in rows
        ],
    )
    return next((m for m in rows if m.session_id == chosen), None)


def build_agent_extras(
    args: argparse.Namespace,
    executor: Any,
    client: Any,
    model: str,
    console: Console,
    scaffold: bool = False,
) -> dict[str, Any]:
    """The optional bits: project instructions and sub-agent delegation."""
    extras: dict[str, Any] = {}

    # Always set, because the schemas one provider requires are the ones
    # another rejects. Whichever provider the user picked has to work.
    extras["tools"] = tools_for(getattr(args, "provider", None) or default_provider())

    if getattr(args, "context", True):
        # Only for a run that acts on the host project. In sandbox mode the
        # agent works in a container and the host directory may not even be
        # the thing being edited.
        # Interactive sessions scaffold on their first prompt instead, so
        # that opening dietcode to ask a question leaves nothing behind.
        if scaffold and not wants_sandbox(args):
            created = ensure_project_context(args.workdir)
            if created:
                console.print(f"[dim]created {created} for this project[/dim]")
        context, source = load_project_context(
                "." if wants_sandbox(args) else args.workdir
            )
        if source:
            extras["system_prompt"] = with_project_context(SYSTEM_PROMPT, context, source)
            console.print(f"[dim]using project instructions from {source}[/dim]")

    handlers: dict[str, Any] = {}

    if getattr(args, "memory", True):
        # The agent's own notes, kept apart from the user's instructions: it
        # may write these, and must not be able to write those.
        project = str(Path(args.workdir).resolve())
        extras["system_prompt"] = with_memory(
            extras.get("system_prompt", SYSTEM_PROMPT), project
        )
        extras["tools"] = [*extras["tools"], REMEMBER_TOOL]
        handlers["remember"] = make_remember_handler(project)

    if getattr(args, "subagents", False):
        extras["tools"] = [*extras["tools"], SPAWN_TOOL]
        handlers["spawn_subagent"] = make_spawn_handler(
            executor, client, model, context_budget=args.context_budget
        )

    if getattr(args, "verify", None):
        # "Done" stops being the model's opinion and becomes this command's
        # exit code.
        extras["verify_command"] = args.verify

    if handlers:
        extras["extra_tool_handlers"] = handlers
    return extras


def run_once(
    args: argparse.Namespace,
    executor: Any,
    client: Any,
    model: str,
    console: Console,
    renderer: Renderer,
    store: SessionStore | None = None,
) -> int:
    started = time.monotonic()
    try:
        result = agent_loop(
            args.task,
            executor,
            client=client,
            model=model,
            max_iterations=args.max_iterations,
            stream=args.stream and not args.quiet,
            context_budget=args.context_budget,
            max_total_tokens=args.max_tokens,
            on_event=None if args.quiet else renderer.on_event,
            **build_agent_extras(args, executor, client, model, console, scaffold=True),
        )
    except KeyboardInterrupt:
        renderer.close()
        console.print("\n[yellow]interrupted[/yellow]")
        return 130
    finally:
        renderer.close()

    # A one-shot run is still a session -- otherwise `dietcode "do a thing"`
    # followed by `dietcode --continue` has nothing to continue from.
    if store is not None:
        store.record_turn(
            result.messages,
            status=result.status,
            steps=result.steps,
            tokens=result.usage.get("total_tokens", 0),
            model=model,
        )

    if args.json:
        print(json.dumps(result.metrics(), indent=2))
    elif args.quiet:
        print(result.summary or result.status)
    else:
        turn_footer(console, result, time.monotonic() - started)
    return 0 if result.ok else 1


def install_signal_handlers(executor: Any) -> None:
    """Tear the container down on SIGTERM as well as on a clean exit.

    SIGKILL cannot be caught, which is exactly why sweep_orphans() exists.
    """

    def handle(signum: int, _frame: Any) -> None:
        executor.close()
        raise SystemExit(128 + signum)

    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is not None:
            try:
                signal.signal(signum, handle)
            except (ValueError, OSError):
                pass  # not on the main thread, or unsupported on this platform


def run_subcommand(argv: list[str], console: Console) -> int:
    args = build_subcommand_parser().parse_args(argv)
    if args.command == "login":
        return login(console, args.provider, args.key)
    if args.command == "logout":
        return logout(console, args.provider)
    if args.command == "doctor":
        return doctor(console)
    return auth_status(console)


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdout()  # before anything prints; Windows consoles default to cp1252

    # .env is a convenience for running from a checkout; installed users have a
    # saved login instead. Optional so the package does not hard-depend on it.
    try:
        from dotenv import find_dotenv, load_dotenv

        # usecwd, because the default searches upward from *this file*. Inside
        # a pipx install that is site-packages, so an installed dietcode never
        # saw the .env sitting in the directory the user was standing in.
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass

    argv = list(sys.argv[1:] if argv is None else argv)
    console = Console()

    # `dietcode login` and friends are dispatched before the agent parser, so a
    # task argument never has to compete with a subcommand name.
    if argv and argv[0] in SUBCOMMANDS:
        return run_subcommand(argv, console)

    args = build_parser().parse_args(argv)

    if args.cleanup:
        removed = DockerExecutor.sweep_orphans(max_age_seconds=0)
        console.print(f"[dim]removed {removed} leftover container(s)[/dim]")
        return 0

    try:
        api_key, base_url, model = resolve_model_config(args)
    except AuthError as exc:
        # Dead-ending on "no credentials" makes the user go read the help, come
        # back, and run a different command. Offer to fix it right here instead.
        if not sys.stdin.isatty():
            console.print(f"[red]{exc}[/red]")
            return 2
        console.print(f"[orange3]{exc}[/orange3]\n")
        if not confirm(console, "Set one up now?"):
            return 2
        if login(console, args.provider, None) != 0:
            return 2
        try:
            api_key, base_url, model = resolve_model_config(args)
        except AuthError as retry_exc:
            console.print(f"[red]{retry_exc}[/red]")
            return 2
        console.print()

    try:
        client = make_client(api_key=api_key, base_url=base_url)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    if wants_sandbox(args):
        # Reap anything a previous crash left behind. Age-based, so a container
        # belonging to another session running right now is left alone.
        stale = DockerExecutor.sweep_orphans()
        if stale:
            console.print(f"[dim]cleaned up {stale} orphaned container(s)[/dim]")

    # Resolved before the executor is built, because the snapshot store is
    # keyed by session id and has to be wrapped around the executor.
    store, history = resolve_session(args, console, model)
    if store is None and history is None and (args.resume is not None or args.fork):
        return 130  # asked to resume something that could not be found

    snapshots = SnapshotStore(
        project=str(Path(args.workdir).resolve()),
        session_id=store.session_id if store else "unsaved",
        enabled=args.snapshots,
    )

    renderer = Renderer(console, show_steps=args.steps)
    try:
        executor, mounts = make_executor(args, console, renderer, snapshots)
    except SandboxError as exc:
        # Docker is optional now that --here exists, so a missing daemon should
        # be a signpost rather than a dead end.
        console.print(f"[red]{exc}[/red]\n")
        console.print("[dim]Either start Docker Desktop, or work without a container:[/dim]")
        console.print("  [bold]dietcode --here[/bold]  [dim]run in this directory, "
                      "asking before each command[/dim]")
        console.print("  [dim]dietcode doctor[/dim]   [dim]check the rest of your setup[/dim]")
        return 2

    install_signal_handlers(executor)
    atexit.register(executor.close)

    try:
        if args.task:
            return run_once(args, executor, client, model, console, renderer, store)

        return Session(
            executor,
            client,
            model=model,
            max_iterations=args.max_iterations,
            mounts=mounts,
            local=not wants_sandbox(args),
            show_steps=args.steps,
            stream=args.stream,
            context_budget=args.context_budget,
            max_total_tokens=args.max_tokens,
            provider=args.provider or default_provider(),
            extras=build_agent_extras(args, executor, client, model, console),
            renderer=renderer,
            store=store,
            history=history,
            snapshots=snapshots,
            scaffold_context=getattr(args, "context", True) and not wants_sandbox(args),
        ).run()
    finally:
        executor.close()


def entrypoint() -> None:
    """Console-script wrapper: turns Ctrl+C into a normal exit code."""
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    entrypoint()
