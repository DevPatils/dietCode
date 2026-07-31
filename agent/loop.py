"""The agent loop.

call model -> execute tool calls -> feed results back -> repeat, until the model
calls task_complete, stops calling tools, or hits max_iterations.

Both entrypoints (cli.py and adapters/terminal_bench.py) call `agent_loop`. The
only thing that differs between them is which Executor gets passed in.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .sandbox import Executor
from .tools import TOOLS, execute_tool, extract_tool_calls_from_text

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Tight by default: Groq's free tier is ~1000 requests/day and a runaway loop
# eats it in one sitting.
DEFAULT_MAX_ITERATIONS = 12

SYSTEM_PROMPT = """You are a command-line coding agent working inside a sandboxed \
Linux container. You complete the user's task by calling tools.

Rules:
- Always invoke tools through the tool-calling API. Never write a call out as \
text in your reply (no <function=...> tags, no JSON describing a call) -- text \
is not executed.
- Work in small steps. Inspect the environment before changing it.
- Prefer run_shell for exploration (ls, cat, grep, find) and for running tests.
- write_file overwrites the whole file, so always pass the complete final contents.
- Verify your work before finishing: re-read files you wrote, run the tests, check \
exit codes.
- When the task is genuinely done and verified, call task_complete with a summary \
of what you did. Do not call it before that.
- If a tool returns an error, read it carefully and correct your next call. Do not \
repeat an identical failing call.

You are graded on the final state of the container, not on your explanation."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0

    def add(self, usage: Any) -> None:
        self.api_calls += 1
        if not usage:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.total_tokens += getattr(usage, "total_tokens", 0) or 0

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
        }


@dataclass
class AgentResult:
    status: str  # complete | stopped | max_iterations_reached | error
    summary: str = ""
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    recovered_tool_calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "complete"

    def metrics(self) -> dict[str, Any]:
        """The numbers that go in the benchmark table."""
        return {
            "status": self.status,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            # How often the model failed to use the tool-calling API at all.
            # Worth reporting separately: it is a model-quality signal, not a
            # scaffold one.
            "recovered_tool_calls": self.recovered_tool_calls,
            **self.usage,
        }


def make_client(api_key: str | None = None, base_url: str = GROQ_BASE_URL) -> Any:
    """Groq speaks the OpenAI protocol, so the OpenAI SDK is the client."""
    from openai import OpenAI

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Put it in .env or export it. "
            "Get a free key at https://console.groq.com/keys"
        )
    return OpenAI(api_key=key, base_url=base_url)


def _tool_call_to_dict(call: Any) -> dict[str, Any]:
    """Normalize an SDK tool call object (or dict) into a plain dict."""
    if isinstance(call, dict):
        fn = call.get("function") or {}
        return {
            "id": call.get("id") or "",
            "type": "function",
            "function": {
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "",
            },
        }
    fn = getattr(call, "function", None)
    return {
        "id": getattr(call, "id", "") or "",
        "type": "function",
        "function": {
            "name": getattr(fn, "name", "") or "",
            "arguments": getattr(fn, "arguments", "") or "",
        },
    }


def _call_llm(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    max_retries: int = 3,
) -> Any:
    """One completion, with backoff on transient failures.

    Rate limits are a normal condition on the free tier, not an error worth
    ending a benchmark task over.
    """
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=list(tools),
                tool_choice="auto",
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises a wide range
            last_exc = exc
            message = str(exc).lower()
            transient = any(
                token in message
                for token in ("rate limit", "429", "timeout", "500", "502", "503", "overloaded")
            )
            if not transient or attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise last_exc  # type: ignore[misc]


def agent_loop(
    task: str,
    executor: Executor,
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    system_prompt: str = SYSTEM_PROMPT,
    tools: Sequence[dict[str, Any]] = TOOLS,
    extra_tool_handlers: dict[str, Callable[[dict[str, Any]], str]] | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> AgentResult:
    """Run the agent until it completes the task, stops, or runs out of steps.

    `extra_tool_handlers` is the hook the stretch-goal spawn_subagent tool plugs
    into -- it needs to recurse into agent_loop, which tools.py must not import.
    """
    client = client or make_client()
    handlers = extra_tool_handlers or {}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    usage = Usage()
    total_tool_calls = 0
    tool_errors = 0
    recovered_tool_calls = 0

    def emit(event: str, **payload: Any) -> None:
        if on_event:
            on_event(event, payload)

    def result(status: str, summary: str, steps: int) -> AgentResult:
        return AgentResult(
            status=status,
            summary=summary,
            steps=steps,
            tool_calls=total_tool_calls,
            tool_errors=tool_errors,
            recovered_tool_calls=recovered_tool_calls,
            usage=usage.as_dict(),
            messages=messages,
        )

    for i in range(max_iterations):
        step = i + 1
        emit("step_start", step=step, max_steps=max_iterations)

        try:
            response = _call_llm(client, model, messages, tools)
        except Exception as exc:  # noqa: BLE001
            emit("error", message=str(exc))
            return result("error", f"LLM call failed: {exc}", step)

        usage.add(getattr(response, "usage", None))
        message = response.choices[0].message
        raw_calls = getattr(message, "tool_calls", None) or []
        calls = [_tool_call_to_dict(c) for c in raw_calls]
        content = getattr(message, "content", None) or ""

        if content:
            emit("assistant_text", step=step, text=content)

        # Open models sometimes write the tool call as prose instead of using
        # the tool_calls field. Without this the run dies on the spot, usually
        # at step 1, and the task scores zero over a formatting slip.
        recovered = False
        if not calls and content:
            text_calls = extract_tool_calls_from_text(content)
            if text_calls:
                calls = [
                    {"id": "", "type": "function", "function": tc} for tc in text_calls
                ]
                recovered = True
                recovered_tool_calls += len(calls)
                emit("recovered_tool_calls", step=step, count=len(calls))

        # Synthesize ids for any call that arrived without one. Every tool_call
        # must be answered by a tool message with a matching id or the next
        # request is rejected by the API.
        for n, call in enumerate(calls):
            if not call["id"]:
                call["id"] = f"call_{step}_{n}"

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            assistant_msg["tool_calls"] = calls
            if recovered:
                # Drop the malformed text now that the call is captured
                # structurally, so the history shows the model one correct
                # example of its own call rather than reinforcing the bad shape.
                assistant_msg["content"] = ""
        messages.append(assistant_msg)

        if not calls:
            # Model stopped acting. Either it thinks it is done without saying
            # so, or it is stuck -- either way there is nothing to feed back.
            emit("stopped", step=step, text=content)
            return result("stopped", content, step)

        tool_messages: list[dict[str, Any]] = []
        for call in calls:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            total_tool_calls += 1
            emit("tool_call", step=step, name=name, arguments=arguments)

            if name == "task_complete":
                from .tools import parse_arguments  # local: keeps the import graph flat

                args, err = parse_arguments(arguments)
                summary = ""
                if not err and args is not None:
                    raw_summary = args.get("summary", "")
                    summary = raw_summary if isinstance(raw_summary, str) else str(raw_summary)
                # Answer the tool call before returning so `messages` stays a
                # valid transcript that can be resumed or logged.
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": summary or "done"}
                )
                emit("complete", step=step, summary=summary)
                return result("complete", summary, step)

            if name in handlers:
                from .tools import parse_arguments

                args, err = parse_arguments(arguments)
                if err or args is None:
                    output = f"Error: {err}"
                else:
                    try:
                        output = handlers[name](args)
                    except Exception as exc:  # noqa: BLE001
                        output = f"Error: {type(exc).__name__}: {exc}"
            else:
                output = execute_tool(name, arguments, executor)

            if output.startswith("Error:"):
                tool_errors += 1
            emit("tool_result", step=step, name=name, output=output)
            tool_messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": output}
            )

        messages.extend(tool_messages)

    emit("max_iterations", step=max_iterations)
    return result("max_iterations_reached", "", max_iterations)
