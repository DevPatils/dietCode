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
    load_project_context,
    make_client,
    with_project_context,
)
from .permissions import PermissionGate, Policy, deny_all
from .prompts import confirm
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
        "--yes",
        action="store_true",
        help="approve every action without asking (dangerous)",
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
    args: argparse.Namespace, console: Console, renderer: Renderer | None = None
) -> tuple[Any, list]:
    if not wants_sandbox(args):
        root = Path(args.workdir).resolve()
        inner = LocalExecutor(root)

        if args.yes:
            console.print(
                f"[{FAIL}] --yes: every command runs without asking [/{FAIL}] "
                f"[orange3]{root}[/orange3]"
            )
            policy = Policy(yes_to_everything=True)
            approver = deny_all  # unreachable while yes_to_everything is set
        elif sys.stdin.isatty():
            policy = Policy()
            approver = make_approver(console, renderer)
        else:
            # Nothing can be asked, so nothing destructive may happen. Silently
            # approving here is how an automated run rewrites someone's files.
            console.print(
                "[orange3]not a terminal: actions that need approval will be "
                "denied. Use --yes to override.[/orange3]"
            )
            policy = Policy()
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
    return executor, mounts


def build_agent_extras(
    args: argparse.Namespace, executor: Any, client: Any, model: str, console: Console
) -> dict[str, Any]:
    """The optional bits: project instructions and sub-agent delegation."""
    extras: dict[str, Any] = {}

    # Always set, because the schemas one provider requires are the ones
    # another rejects. Whichever provider the user picked has to work.
    extras["tools"] = tools_for(getattr(args, "provider", None) or default_provider())

    if getattr(args, "context", True):
        context, source = load_project_context(
                "." if wants_sandbox(args) else args.workdir
            )
        if source:
            extras["system_prompt"] = with_project_context(SYSTEM_PROMPT, context, source)
            console.print(f"[dim]using project instructions from {source}[/dim]")

    if getattr(args, "subagents", False):
        extras["tools"] = [*extras["tools"], SPAWN_TOOL]
        extras["extra_tool_handlers"] = {
            "spawn_subagent": make_spawn_handler(
                executor, client, model, context_budget=args.context_budget
            )
        }
    return extras


def run_once(
    args: argparse.Namespace,
    executor: Any,
    client: Any,
    model: str,
    console: Console,
    renderer: Renderer,
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
            **build_agent_extras(args, executor, client, model, console),
        )
    except KeyboardInterrupt:
        renderer.close()
        console.print("\n[yellow]interrupted[/yellow]")
        return 130
    finally:
        renderer.close()

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

    renderer = Renderer(console, show_steps=args.steps)
    try:
        executor, mounts = make_executor(args, console, renderer)
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
            return run_once(args, executor, client, model, console, renderer)
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
