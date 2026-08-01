"""Entrypoint.

    python cli.py                     interactive session
    python cli.py "do the thing"      one shot, then exit

Both run the same agent_loop against the same sandbox; the interactive path
just keeps the container and the conversation alive between turns.
"""

from __future__ import annotations

import argparse
import atexit
import json
import signal
import sys
import time
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from agent.loop import (
    DEFAULT_CONTEXT_BUDGET,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MODEL,
    agent_loop,
    make_client,
)
from agent.repl import Session
from agent.sandbox import (
    DEFAULT_CPUS,
    DEFAULT_IMAGE,
    DEFAULT_MEMORY,
    DEFAULT_PIDS_LIMIT,
    DockerExecutor,
    LocalExecutor,
    SandboxError,
)
from agent.ui import Renderer, turn_footer, use_utf8_stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI coding agent. Run with no task for an interactive session."
    )
    parser.add_argument("task", nargs="?", help="what to do; omit for interactive mode")
    parser.add_argument(
        "--mount",
        action="append",
        metavar="HOSTDIR[:TARGET]",
        help="bind-mount a host directory into the sandbox so files persist "
        "(default target /workspace). Repeatable.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="run on the host instead of Docker (no isolation, dev only)",
    )
    parser.add_argument("--workdir", default=".", help="working dir for --local")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="sandbox container image")
    parser.add_argument(
        "--container", help="attach to an existing container instead of creating one"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="stop once this many tokens have been spent (per task)",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=DEFAULT_CONTEXT_BUDGET,
        metavar="N",
        help=f"trim the oldest turns above this prompt size (default {DEFAULT_CONTEXT_BUDGET})",
    )
    parser.add_argument("--memory", default=DEFAULT_MEMORY, help="container memory cap")
    parser.add_argument(
        "--cpus", default=DEFAULT_CPUS, help="container CPU cap"
    )
    parser.add_argument(
        "--pids-limit", type=int, default=DEFAULT_PIDS_LIMIT, help="container process cap"
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="cut the sandbox off from the network entirely",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="remove every leftover agent container and exit",
    )
    parser.add_argument("--steps", action="store_true", help="show step separators")
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="wait for each reply instead of showing it as it is generated",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the final result")
    parser.add_argument("--json", action="store_true", help="print metrics as JSON")
    return parser


def make_executor(args: argparse.Namespace, console: Console) -> tuple[Any, list]:
    if args.local:
        console.print(
            "[orange3]running unsandboxed on the host: the model's shell commands "
            "run as you[/orange3]"
        )
        return LocalExecutor(args.workdir), []

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


def run_once(args: argparse.Namespace, executor: Any, client: Any) -> int:
    console = Console(quiet=args.quiet and not args.json)
    renderer = Renderer(console, show_steps=args.steps)
    started = time.monotonic()
    try:
        result = agent_loop(
            args.task,
            executor,
            client=client,
            model=args.model,
            max_iterations=args.max_iterations,
            stream=args.stream and not args.quiet,
            context_budget=args.context_budget,
            max_total_tokens=args.max_tokens,
            on_event=None if args.quiet else renderer.on_event,
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


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdout()  # before anything prints; Windows consoles default to cp1252
    load_dotenv()
    args = build_parser().parse_args(argv)
    console = Console()

    if args.cleanup:
        removed = DockerExecutor.sweep_orphans(max_age_seconds=0)
        console.print(f"[dim]removed {removed} leftover container(s)[/dim]")
        return 0

    try:
        client = make_client()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    if not args.local:
        # Reap anything a previous crash left behind. Age-based, so a container
        # belonging to another session running right now is left alone.
        stale = DockerExecutor.sweep_orphans()
        if stale:
            console.print(f"[dim]cleaned up {stale} orphaned container(s)[/dim]")

    try:
        executor, mounts = make_executor(args, console)
    except SandboxError as exc:
        console.print(f"[red]sandbox error: {exc}[/red]")
        console.print("[dim]is Docker running? or use --local to run on the host[/dim]")
        return 2

    install_signal_handlers(executor)
    atexit.register(executor.close)

    try:
        if args.task:
            return run_once(args, executor, client)
        return Session(
            executor,
            client,
            model=args.model,
            max_iterations=args.max_iterations,
            mounts=mounts,
            local=args.local,
            show_steps=args.steps,
            stream=args.stream,
            context_budget=args.context_budget,
            max_total_tokens=args.max_tokens,
        ).run()
    finally:
        executor.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
