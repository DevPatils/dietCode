"""Sub-agent delegation.

A fresh `agent_loop` with its own message history, returning **only** its final
summary to the parent. The isolation is the entire mechanism being tested: if
the child's transcript came back, the parent's context would grow exactly as
fast as doing the work inline, and there would be nothing to measure.

Lives outside tools.py because it has to call agent_loop, and tools.py must
not import the loop -- that cycle is why `extra_tool_handlers` exists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .sandbox import Executor
from .tools import TOOLS

SPAWN_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "spawn_subagent",
        "description": (
            "Delegate a self-contained piece of work to a fresh agent that "
            "shares your files but not your conversation. It reports back only "
            "a summary. Use it for work whose details you do not need to keep "
            "in mind -- surveying a large codebase, or a chunk of a refactor. "
            "Do not use it for a task you could finish in a step or two: it "
            "costs a whole extra conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "A complete, self-contained instruction. The sub-agent "
                        "cannot see your conversation, so include every detail "
                        "it needs, including exact paths."
                    ),
                },
            },
            "required": ["task"],
        },
    },
}

# A sub-agent gets a shorter leash than its parent: the point is a bounded
# errand, and an unbounded child could spend the whole run's budget alone.
SUBAGENT_MAX_ITERATIONS = 8

SUBAGENT_PROMPT = """You are a sub-agent working inside a larger task. You have \
the same tools and the same files, but you cannot see the parent's conversation \
and it cannot see yours.

Do exactly the task you were given, then call task_complete with a summary that \
stands on its own: what you changed, what you found, which paths are involved, \
and anything that did not work. That summary is the only thing the parent \
receives, so leave nothing important out of it."""


def make_spawn_handler(
    executor: Executor,
    client: Any,
    model: str,
    *,
    max_iterations: int = SUBAGENT_MAX_ITERATIONS,
    context_budget: int | None = None,
    depth: int = 0,
    max_depth: int = 1,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
    tools: Sequence[dict[str, Any]] = TOOLS,
) -> Callable[[dict[str, Any]], str]:
    """Build the `spawn_subagent` handler for `extra_tool_handlers`.

    `max_depth` is a hard stop: a sub-agent that can spawn sub-agents can
    recurse until the request quota is gone, and nothing in the model's
    behaviour reliably prevents it.
    """
    from .loop import agent_loop  # local import: tools.py must not pull in the loop

    def spawn(args: dict[str, Any]) -> str:
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            return "Error: spawn_subagent needs a 'task' string describing the work."

        if depth >= max_depth:
            return (
                "Error: sub-agents cannot spawn further sub-agents. "
                "Do this part of the work yourself."
            )

        if on_event:
            on_event("subagent_start", {"task": task, "depth": depth + 1})

        # No spawn tool in the child's toolset, and no history: this is the
        # isolation the whole feature exists to test.
        child = agent_loop(
            task,
            executor,
            client=client,
            model=model,
            max_iterations=max_iterations,
            system_prompt=SUBAGENT_PROMPT,
            tools=tools,
            **({"context_budget": context_budget} if context_budget else {}),
        )

        if on_event:
            on_event(
                "subagent_done",
                {
                    "status": child.status,
                    "steps": child.steps,
                    "tokens": child.usage.get("total_tokens", 0),
                    "depth": depth + 1,
                },
            )

        summary = (child.summary or "").strip()
        if child.status == "complete" and summary:
            return summary
        if summary:
            # Say how it ended, so the parent does not read a partial result as
            # a finished one.
            return f"[sub-agent {child.status} after {child.steps} steps] {summary}"
        return (
            f"[sub-agent {child.status} after {child.steps} steps with no summary] "
            f"Treat this as unfinished and check the files yourself."
        )

    return spawn
