"""The agent loop.

call model -> execute tool calls -> feed results back -> repeat, until the model
calls task_complete, stops calling tools, or hits max_iterations.

Both entrypoints (cli.py and adapters/terminal_bench.py) call `agent_loop`. The
only thing that differs between them is which Executor gets passed in.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .sandbox import Executor
from .tools import TOOLS, execute_tool, extract_tool_calls_from_text

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Tight by default: Groq's free tier is ~1000 requests/day and a runaway loop
# eats it in one sitting.
DEFAULT_MAX_ITERATIONS = 12

# Per-request ceiling. Long enough for a slow generation, short enough that a
# wedged connection does not eat a benchmark task's whole time budget.
REQUEST_TIMEOUT = float(os.environ.get("AGENT_REQUEST_TIMEOUT", "120"))

# Connection blips are common when Docker is saturating the network pulling task
# images -- observed live, two benchmark tasks died on the first call with zero
# tokens spent. Four attempts with a longer backoff rides those out.
MAX_LLM_RETRIES = 4

# How many times to bounce a task_complete that was batched with the work it
# claims to have verified. Bounded so a model that always batches still
# terminates instead of burning the daily request quota.
MAX_COMPLETION_DEFERRALS = 2

# Prompt tokens allowed before the oldest turns get dropped. Well under
# llama-3.3-70b's 128k window: the ceiling that bites first is Groq's
# tokens-per-minute limit, not the model's context.
DEFAULT_CONTEXT_BUDGET = int(os.environ.get("AGENT_CONTEXT_BUDGET", "48000"))

SYSTEM_PROMPT = """You are a command-line coding agent working inside a sandboxed \
Linux container. You complete the user's task by calling tools.

Rules:
- Always invoke tools through the tool-calling API. Never write a call out as \
text in your reply (no <function=...> tags, no JSON describing a call) -- text \
is not executed.
- Work in small steps. Inspect the environment before changing it.
- Use find_files to locate files and search to find code. They are cheaper and \
more reliable than shelling out to find or grep.
- To change an existing file use edit_file, which replaces an exact snippet. \
Only use write_file to create a new file or replace one outright: it rewrites \
everything, which is expensive and easy to get wrong.
- edit_file needs `old` to match the file exactly, whitespace included. If it \
reports no match, read the file again rather than guessing.
- Use run_shell to run commands and tests.
- Verify your work before finishing: re-read files you wrote, run the tests, check \
exit codes.
- When the task is genuinely done and verified, call task_complete with a summary \
of what you did. Do not call it before that.
- If a tool returns an error, read it carefully and correct your next call. Do not \
repeat an identical failing call.

You are graded on the final state of the container, not on your explanation."""

# Files a project can leave for the agent, in the order they are looked for.
# AGENTS.md is the emerging cross-tool convention; the others are what people
# already have lying around.
CONTEXT_FILES = ("DIETCODE.md", "AGENTS.md", "CLAUDE.md", ".cursorrules")
MAX_CONTEXT_CHARS = 8000


def load_project_context(root: str | os.PathLike[str] = ".") -> tuple[str, str | None]:
    """Read a project's instructions file, if it has one.

    Returns (text, filename). Read from the host rather than through the
    Executor on purpose: this is the *user's* standing instructions, and it
    should not be something the agent can rewrite mid-run to change its own
    rules.
    """
    from pathlib import Path

    base = Path(root)
    for name in CONTEXT_FILES:
        candidate = base / name
        try:
            if not candidate.is_file():
                continue
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        if len(text) > MAX_CONTEXT_CHARS:
            text = text[:MAX_CONTEXT_CHARS] + "\n… [truncated]"
        return text, name
    return "", None


def with_project_context(system_prompt: str, context: str, source: str | None) -> str:
    """Append the project's own instructions, marked as outranking the defaults."""
    if not context:
        return system_prompt
    return (
        f"{system_prompt}\n\n"
        f"--- Project instructions from {source} ---\n"
        f"These come from the person you are working for. Where they conflict "
        f"with anything above, follow these.\n\n"
        f"{context}"
    )


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
    # complete | stopped | max_iterations_reached | budget_exhausted | error
    status: str
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


def make_client(
    api_key: str | None = None,
    base_url: str = GROQ_BASE_URL,
    timeout: float = REQUEST_TIMEOUT,
) -> Any:
    """Every supported provider speaks the OpenAI protocol, so one client and a
    different base_url covers all of them."""
    from openai import OpenAI

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key. Run `dietcode login` to save one, "
            "or set an API key environment variable."
        )
    return OpenAI(
        api_key=key,
        base_url=base_url,
        # The SDK defaults to a 600s timeout, so one hung request can stall a
        # benchmark task for ten minutes and then fail anyway.
        timeout=timeout,
        # Retries are ours: _call_llm classifies the error and backs off, and
        # the SDK retrying underneath would multiply the wait invisibly.
        max_retries=0,
    )


# Fields a provider attaches to a tool call that are not part of the OpenAI
# schema but must survive the round trip. Gemini 3 returns
# extra_content.google.thought_signature and rejects the next request with a
# 400 if it is not echoed back -- rebuilding the assistant turn from just
# id/type/function silently drops it and every tool-using conversation dies on
# its second step.
def _passthrough_fields(call: Any) -> dict[str, Any]:
    if isinstance(call, dict):
        return {
            k: v for k, v in call.items() if k not in ("id", "type", "function")
        }
    return dict(getattr(call, "model_extra", None) or {})


def _tool_call_to_dict(call: Any) -> dict[str, Any]:
    """Normalize an SDK tool call object (or dict) into a plain dict."""
    if isinstance(call, dict):
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        arguments = fn.get("arguments") or ""
        call_id = call.get("id") or ""
    else:
        fn = getattr(call, "function", None)
        name = getattr(fn, "name", "") or ""
        arguments = getattr(fn, "arguments", "") or ""
        call_id = getattr(call, "id", "") or ""

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
        **_passthrough_fields(call),
    }


def failed_generation_text(exc: Exception) -> str | None:
    """Pull the rejected text out of a Groq `tool_use_failed` 400.

    Groq validates tool arguments against our schema server-side. When a model
    sends the wrong type -- `"timeout": "10"` instead of `10` -- the request is
    rejected outright and the run dies, even though our dispatcher would have
    coerced it happily. The error body carries the generation it refused, so we
    can recover the call from it instead of losing the task.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    # The wire format is {"error": {...}}, but openai>=2 unwraps it and exposes
    # the inner dict as .body. Accept either, since assuming one shape silently
    # disables this whole path -- which is exactly what happened.
    inner = body.get("error")
    error = inner if isinstance(inner, dict) else body
    if error.get("code") != "tool_use_failed":
        return None
    failed = error.get("failed_generation")
    return failed if isinstance(failed, str) and failed.strip() else None


def estimate_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """Rough token count for trimming decisions.

    Deliberately not tiktoken: that is OpenAI's tokenizer, not Llama's, so it
    would be precisely wrong rather than approximately right -- and this only
    needs to decide *whether* to trim, not to bill anyone. ~4 chars per token
    plus per-message overhead.
    """
    total = 0
    for message in messages:
        total += 4  # role and framing
        content = message.get("content") or ""
        total += len(content) // 4
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            total += len(function.get("name") or "") // 4
            total += len(function.get("arguments") or "") // 4
            total += 8
    return total


def _blocks(messages: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group messages into units that must be kept or dropped together.

    An assistant message with tool_calls and the tool messages answering it are
    atomic: dropping half a pair leaves a tool_call with no result (or a result
    with no call), and the API rejects the whole request.
    """
    grouped: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        block = [message]
        index += 1
        if message.get("role") == "assistant" and message.get("tool_calls"):
            while index < len(messages) and messages[index].get("role") == "tool":
                block.append(messages[index])
                index += 1
        grouped.append(block)
    return grouped


def trim_messages(
    messages: Sequence[dict[str, Any]], budget: int
) -> tuple[list[dict[str, Any]], int]:
    """Drop the oldest turns until the transcript fits. Returns (messages, dropped).

    Without this an interactive session grows until it exceeds the model's
    context window, after which *every* subsequent request fails and the session
    is unrecoverable.
    """
    messages = list(messages)
    if estimate_tokens(messages) <= budget:
        return messages, 0

    system: list[dict[str, Any]] = []
    rest = messages
    if messages and messages[0].get("role") == "system":
        system, rest = [messages[0]], messages[1:]

    blocks = _blocks(rest)
    kept: list[list[dict[str, Any]]] = []
    total = estimate_tokens(system)

    # Newest first: recent context is what the model needs to keep working.
    for block in reversed(blocks):
        cost = estimate_tokens(block)
        if kept and total + cost > budget:
            break
        kept.insert(0, block)
        total += cost

    # A leading tool message would be an orphan once its assistant turn is gone.
    while kept and kept[0] and kept[0][0].get("role") == "tool":
        kept.pop(0)

    dropped_blocks = len(blocks) - len(kept)
    if dropped_blocks <= 0:
        return messages, 0

    dropped_messages = sum(len(b) for b in blocks[: len(blocks) - len(kept)])
    note = [
        {
            "role": "user",
            "content": (
                f"[{dropped_messages} earlier messages were dropped to stay within "
                f"the context limit. Re-read any file you need rather than relying "
                f"on memory of it.]"
            ),
        }
    ]
    return [*system, *note, *[m for block in kept for m in block]], dropped_messages


def is_context_error(exc: Exception) -> bool:
    """Whether the request failed because the transcript no longer fits."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        error = inner if isinstance(inner, dict) else body
        if error.get("code") == "context_length_exceeded":
            return True
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "context_length_exceeded",
            "context length",
            "maximum context",
            "too many tokens",
            "reduce the length",
        )
    )


def is_quota_exhausted(exc: Exception) -> bool:
    """A daily cap, as opposed to a per-minute burst limit.

    Both arrive as 429, but they need opposite handling: a per-minute limit
    clears in seconds and is worth backing off for, while a tokens-per-day cap
    will not clear for hours. Retrying the latter just burns the clock and then
    fails anyway -- which is exactly what happened to a benchmark run.
    """
    text = str(exc).lower()
    if "per day" in text or "tpd" in text or "rpd" in text:
        return True
    # Gemini words it differently and never says "per day" in the message --
    # the daily part is only in the quotaId. Missing it cost 22 seconds of
    # backoff per turn against a cap that does not clear until tomorrow.
    return "exceeded your current quota" in text and "perday" in text.replace("-", "")


def _is_transient(exc: Exception) -> bool:
    """Whether retrying could plausibly help.

    Prefers the SDK's typed exceptions and HTTP status over grepping the message
    text -- string-matching an error body is how the tool_use_failed bug hid.
    The text check stays as a fallback for non-SDK clients.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600

    try:
        import openai

        if isinstance(
            exc,
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ),
        ):
            return True
        if isinstance(exc, openai.APIStatusError):
            return False  # a typed 4xx: retrying sends the same bad request
    except ImportError:
        pass

    text = str(exc).lower()
    return any(
        token in text
        for token in ("rate limit", "429", "timeout", "500", "502", "503", "overloaded")
    )


@dataclass
class Completion:
    """One model turn, however it arrived.

    Streaming and non-streaming are normalized to this so the loop never has to
    care which transport produced the turn.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Any = None


def _from_response(response: Any) -> Completion:
    message = response.choices[0].message
    raw_calls = getattr(message, "tool_calls", None) or []
    return Completion(
        content=getattr(message, "content", None) or "",
        tool_calls=[_tool_call_to_dict(c) for c in raw_calls],
        usage=getattr(response, "usage", None),
    )


def _from_stream(stream: Any, on_text: Callable[[str], None] | None) -> Completion:
    """Reassemble a streamed turn.

    Tool calls arrive in fragments: the id and name usually land in the first
    delta for that index, then the arguments accumulate across many. They are
    keyed by index because several calls stream interleaved.
    """
    parts: list[str] = []
    slots: dict[int, dict[str, str]] = {}
    extras: dict[int, dict[str, Any]] = {}
    usage = None

    for chunk in stream:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage  # arrives in a final choice-less chunk
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        text = getattr(delta, "content", None)
        if text:
            parts.append(text)
            if on_text is not None:
                on_text(text)

        for raw in getattr(delta, "tool_calls", None) or []:
            index = getattr(raw, "index", None)
            if index is None:
                index = len(slots)
            slot = slots.setdefault(
                index, {"id": "", "name": "", "arguments": "", "_last_name": ""}
            )
            # Provider-specific fields (Gemini's thought_signature) arrive on
            # whichever fragment carries them and must survive reassembly too.
            extras.setdefault(index, {}).update(_passthrough_fields(raw))
            call_id = getattr(raw, "id", None)
            if call_id:
                slot["id"] = call_id
            function = getattr(raw, "function", None)
            if function is None:
                continue
            name_delta = getattr(function, "name", None)
            # Most providers send the name complete in the first delta; a few
            # repeat it on every one. Accumulate so a genuinely chunked name
            # survives, but skip an immediate repeat so it is not doubled.
            if name_delta and name_delta != slot["_last_name"]:
                slot["name"] += name_delta
                slot["_last_name"] = name_delta
            args_delta = getattr(function, "arguments", None)
            if args_delta:
                slot["arguments"] += args_delta

    calls = [
        {
            "id": slot["id"],
            "type": "function",
            "function": {"name": slot["name"], "arguments": slot["arguments"]},
            **extras.get(index, {}),
        }
        for index, slot in sorted(slots.items())
    ]
    return Completion(content="".join(parts), tool_calls=calls, usage=usage)


def _call_llm(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    *,
    stream: bool = False,
    on_text: Callable[[str], None] | None = None,
    max_retries: int = MAX_LLM_RETRIES,
) -> Completion:
    """One completion, with backoff on transient failures.

    Rate limits and connection blips are normal conditions here, not errors
    worth ending a benchmark task over.
    """
    delay = 3.0
    last_exc: Exception | None = None
    include_usage = True

    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": list(tools),
                "tool_choice": "auto",
            }
            if stream:
                kwargs["stream"] = True
                if include_usage:
                    # Without this a streamed turn reports no token usage at
                    # all, which would blank the benchmark's token column.
                    kwargs["stream_options"] = {"include_usage": True}
            raw = client.chat.completions.create(**kwargs)
            return _from_stream(raw, on_text) if stream else _from_response(raw)

        except Exception as exc:  # noqa: BLE001 - SDK raises a wide range
            last_exc = exc

            rejected = failed_generation_text(exc)
            if rejected:
                return Completion(content=rejected)

            # Not every OpenAI-compatible server knows stream_options. Losing
            # the token count is much better than losing the turn.
            if include_usage and "stream_options" in str(exc).lower():
                include_usage = False
                continue

            # A daily cap will not clear during this run; fail immediately so
            # the caller sees the reason instead of a timeout.
            if is_quota_exhausted(exc):
                raise

            if not _is_transient(exc) or attempt == max_retries - 1:
                raise
            # A stream that failed part-way has already shown the user some
            # text; retrying will repeat it. Rare enough to accept.
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
    stream: bool = False,
    context_budget: int = DEFAULT_CONTEXT_BUDGET,
    max_total_tokens: int | None = None,
    history: Sequence[dict[str, Any]] | None = None,
    extra_tool_handlers: dict[str, Callable[[dict[str, Any]], str]] | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> AgentResult:
    """Run the agent until it completes the task, stops, or runs out of steps.

    `stream` emits the reply token by token as `assistant_delta` events. It
    defaults off: the benchmark has no console to stream to, and reassembling
    tool calls from deltas is strictly more machinery to go wrong, so scored
    runs take the simpler path. Human-facing callers turn it on.

    `history` continues an earlier conversation -- pass a previous result's
    `.messages` to give the agent memory of what it already did. It is copied,
    not mutated, so an interrupted turn cannot leave the caller holding a
    transcript with unanswered tool calls in it.

    `extra_tool_handlers` is the hook the stretch-goal spawn_subagent tool plugs
    into -- it needs to recurse into agent_loop, which tools.py must not import.
    """
    client = client or make_client()
    handlers = extra_tool_handlers or {}
    if history:
        messages = [dict(m) for m in history]
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": task})
    usage = Usage()
    total_tool_calls = 0
    tool_errors = 0
    recovered_tool_calls = 0
    deferrals = 0
    last_summary = ""

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

        completion = None
        for context_attempt in range(2):
            sent, dropped = trim_messages(messages, context_budget)
            if dropped:
                emit("context_trimmed", step=step, dropped=dropped, budget=context_budget)
            try:
                completion = _call_llm(
                    client,
                    model,
                    sent,
                    tools,
                    stream=stream,
                    # step is bound as a default: a bare closure over the loop
                    # variable would report whatever step the loop had reached
                    # by the time the callback fired.
                    on_text=(
                        (lambda text, at=step: emit("assistant_delta", step=at, text=text))
                        if stream
                        else None
                    ),
                )
                break
            except Exception as exc:  # noqa: BLE001
                # Our estimate is approximate, so the server can still say no.
                # Shrink relative to what was actually sent, not to the budget:
                # if the budget is far above the real size, halving it changes
                # nothing and the retry sends the identical payload.
                if context_attempt == 0 and is_context_error(exc):
                    context_budget = max(2_000, estimate_tokens(sent) // 2)
                    emit("context_trimmed", step=step, dropped=0, budget=context_budget)
                    continue
                emit("error", message=str(exc))
                return result("error", f"LLM call failed: {exc}", step)

        if completion is None:  # pragma: no cover - loop always sets or returns
            return result("error", "LLM call failed", step)

        usage.add(completion.usage)
        calls = completion.tool_calls
        content = completion.content

        if max_total_tokens and usage.total_tokens >= max_total_tokens:
            # A hard spend ceiling. max_iterations bounds steps, not tokens, and
            # a single step with a large tool result can be enormous.
            emit("budget_exhausted", step=step, tokens=usage.total_tokens)
            return result("budget_exhausted", content, step)

        # When streaming, the text has already been delivered delta by delta.
        if content and not stream:
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
        completion: tuple[dict[str, Any], str] | None = None

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
                # Do not return here. Models batch task_complete together with
                # the work in a single turn, and returning mid-batch would skip
                # every call after it and leave their tool_call ids unanswered
                # in the transcript.
                completion = (call, summary)
                tool_messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": summary or "done"}
                )
                continue

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

        if completion is not None:
            call, summary = completion
            # If task_complete arrived alone, the model has already seen the
            # results it is judging and we take it at its word.
            if len(calls) == 1 or deferrals >= MAX_COMPLETION_DEFERRALS:
                emit("complete", step=step, summary=summary)
                return result("complete", summary, step)

            # Otherwise it declared success in the same breath as the work,
            # before any of those results existed. Hand back the results and
            # make it say so again. Bounded, so a model that always batches
            # still finishes rather than burning the request quota.
            deferrals += 1
            last_summary = summary
            for msg in tool_messages:
                if msg["tool_call_id"] == call["id"]:
                    msg["content"] = (
                        "Not recorded yet: you called task_complete in the same "
                        "turn as the tool calls above, so you had not seen their "
                        "results when you claimed the task was done. Review the "
                        "results, then call task_complete on its own if the task "
                        "really is finished."
                    )
            emit("completion_deferred", step=step, summary=summary)

    emit("max_iterations", step=max_iterations)
    return result("max_iterations_reached", last_summary, max_iterations)
