"""Entrypoint: python cli.py "task description"

Runs the agent against a Docker sandbox it creates and tears down. Use
--local to skip Docker and run on the host filesystem (development only --
the agent can run arbitrary commands as you).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from dotenv import load_dotenv

from agent.loop import DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL, agent_loop, make_client
from agent.sandbox import DEFAULT_IMAGE, DockerExecutor, LocalExecutor, SandboxError

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
GREEN, RED, YELLOW = "\033[32m", "\033[31m", "\033[33m"


def make_printer(quiet: bool) -> Any:
    def on_event(event: str, payload: dict[str, Any]) -> None:
        if quiet:
            return
        if event == "step_start":
            print(f"\n{DIM}--- step {payload['step']}/{payload['max_steps']} ---{RESET}")
        elif event == "assistant_text":
            print(f"{payload['text']}")
        elif event == "tool_call":
            args = payload["arguments"]
            if len(args) > 200:
                args = args[:200] + "..."
            print(f"{BOLD}> {payload['name']}{RESET} {DIM}{args}{RESET}")
        elif event == "tool_result":
            output = payload["output"]
            colour = RED if output.startswith("Error:") else DIM
            if len(output) > 500:
                output = output[:500] + f"\n{DIM}... [truncated for display]{RESET}"
            print(f"{colour}{output}{RESET}")
        elif event == "recovered_tool_calls":
            print(
                f"{YELLOW}(model wrote {payload['count']} tool call(s) as text; "
                f"recovered){RESET}"
            )
        elif event == "complete":
            print(f"\n{GREEN}task_complete{RESET}: {payload['summary']}")
        elif event == "stopped":
            print(f"\n{YELLOW}model stopped without calling a tool{RESET}")
        elif event == "max_iterations":
            print(f"\n{RED}hit max iterations ({payload['step']}){RESET}")
        elif event == "error":
            print(f"\n{RED}error: {payload['message']}{RESET}")

    return on_event


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="CLI coding agent")
    parser.add_argument("task", help="what the agent should do")
    parser.add_argument(
        "--local",
        action="store_true",
        help="run on the host instead of Docker (no isolation, dev only)",
    )
    parser.add_argument("--workdir", default=".", help="working dir for --local")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="sandbox container image")
    parser.add_argument("--container", help="attach to an existing container instead of creating one")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--quiet", action="store_true", help="only print the final result")
    parser.add_argument("--json", action="store_true", help="print result metrics as JSON")
    args = parser.parse_args(argv)

    try:
        client = make_client()
    except RuntimeError as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 2

    try:
        if args.local:
            executor: Any = LocalExecutor(args.workdir)
            if not args.quiet:
                print(f"{YELLOW}running unsandboxed on the host in {args.workdir}{RESET}")
        else:
            executor = DockerExecutor(image=args.image, container=args.container)
            if not args.quiet:
                print(f"{DIM}sandbox: {executor.container}{RESET}")
    except SandboxError as exc:
        print(f"{RED}sandbox error: {exc}{RESET}", file=sys.stderr)
        print(f"{DIM}is Docker running? or use --local for host execution{RESET}", file=sys.stderr)
        return 2

    try:
        result = agent_loop(
            args.task,
            executor,
            client=client,
            model=args.model,
            max_iterations=args.max_iterations,
            on_event=make_printer(args.quiet),
        )
    finally:
        executor.close()

    if args.json:
        print(json.dumps(result.metrics(), indent=2))
    elif args.quiet:
        print(result.summary or result.status)
    else:
        m = result.metrics()
        recovered = (
            f", {m['recovered_tool_calls']} recovered from text"
            if m["recovered_tool_calls"]
            else ""
        )
        print(
            f"\n{DIM}{m['status']} in {m['steps']} steps, {m['tool_calls']} tool calls "
            f"({m['tool_errors']} errors{recovered}), {m.get('total_tokens', 0)} tokens{RESET}"
        )

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
