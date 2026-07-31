"""Tool schemas and dispatch.

The contract for `execute_tool` is that it NEVER raises. Llama and Qwen produce
malformed tool-call JSON, invented tool names, and wrong-typed arguments often
enough that treating those as exceptions would kill the loop several times per
benchmark run. Every failure comes back as a plain error string in the tool
result so the model can see what it did wrong and correct on the next turn.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .sandbox import DEFAULT_TIMEOUT, Executor, SandboxError

# Cap on a single tool result. Shell commands like `find /` can emit megabytes,
# which would blow the context window and the free-tier token budget in one turn.
MAX_TOOL_RESULT_CHARS = 8000

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the full text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file, creating it if needed and overwriting it "
                "if it exists. Always pass the complete final contents of the file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "content": {"type": "string", "description": "Full file contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command and return its stdout, stderr and exit code. "
                "Runs in a sandboxed container. Non-interactive only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run."},
                    "timeout": {
                        # Deliberately accepts a string too. Groq validates tool
                        # arguments against this schema server-side and rejects
                        # the whole generation with a 400 on a mismatch -- and
                        # models send "10" as often as 10. _coerce_timeout
                        # normalizes it, so strictness here buys nothing and
                        # costs whole runs.
                        "type": ["integer", "string"],
                        "description": f"Seconds before the command is killed (default {DEFAULT_TIMEOUT}).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": (
                "Call this when the task is fully done and verified. This ends the "
                "session, so do not call it speculatively."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "What you did and how you verified it.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]


# --- recovering tool calls the model wrote as prose -------------------------
#
# Llama and Qwen intermittently emit a tool call as *text* in the content field
# instead of through the API's tool_calls field -- e.g.
#     <function/run_shell {"command": "ls"} </function>
# Groq's server-side parser catches most of these, but not all. When one slips
# through, the loop sees a turn with no tool calls and stops dead on step 1, so
# recovering them is the difference between a task running and a task scoring 0.

# Matches <function=name>, <function/name, <function name>, <function: name>.
_FUNCTION_TAG_RE = re.compile(r"<function\s*[=/:]?\s*([A-Za-z_][\w.-]*)\s*>?", re.IGNORECASE)

# Llama's native tool token, which precedes a bare JSON object.
_PYTHON_TAG_RE = re.compile(r"<\|python_tag\|>")

_FENCE_RE = re.compile(r"```(?:json|python|tool_code)?", re.IGNORECASE)


def _decode_object_at(text: str, index: int, window: int = 200) -> dict[str, Any] | None:
    """Decode the first JSON object at/after `index`.

    Uses raw_decode rather than a regex so that braces inside string values --
    which file content is full of -- do not truncate the match.
    """
    brace = text.find("{", index, index + window)
    if brace == -1:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(text, brace)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _iter_json_objects(text: str):
    """Yield every top-level JSON object in the text, in order."""
    decoder = json.JSONDecoder()
    i = 0
    while True:
        i = text.find("{", i)
        if i == -1:
            return
        try:
            obj, end = decoder.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            yield obj
        i = max(end, i + 1)


def _normalize(name: Any, args: Any) -> dict[str, str] | None:
    """Accept a recovered call only if it names a tool we actually have --
    otherwise prose that merely mentions a tool would be executed."""
    if not isinstance(name, str) or name not in TOOL_NAMES:
        return None
    if isinstance(args, str):
        return {"name": name, "arguments": args}
    if isinstance(args, dict):
        return {"name": name, "arguments": json.dumps(args)}
    return None


def extract_tool_calls_from_text(content: Any) -> list[dict[str, str]]:
    """Recover tool calls the model wrote as text. Returns [] if there are none."""
    if not isinstance(content, str) or "{" not in content:
        return []

    text = _FENCE_RE.sub("", content)
    found: list[dict[str, str]] = []

    # 1. Explicit function tags -- the tag names the tool, the object is its args.
    for match in _FUNCTION_TAG_RE.finditer(text):
        obj = _decode_object_at(text, match.end())
        if obj is not None:
            call = _normalize(match.group(1), obj)
            if call:
                found.append(call)
    if found:
        return found

    # 2. Bare JSON objects carrying their own name, with or without a python_tag.
    #    {"name": "run_shell", "parameters": {...}} / {..., "arguments": {...}}
    for obj in _iter_json_objects(text):
        args = obj.get("parameters", obj.get("arguments"))
        call = _normalize(obj.get("name"), {} if args is None else args)
        if call:
            found.append(call)
    if found:
        return found

    # 3. A python_tag followed by an object with no name -- unrecoverable, since
    #    we cannot tell which tool was meant. Left to the caller as "stopped".
    return []


def truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Keep the head and tail. Errors usually land at the end of output, so a
    plain head-only cut tends to discard the useful part."""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n\n... [{omitted} characters omitted] ...\n\n{text[-tail:]}"


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Coerce whatever the model sent into a dict. Returns (args, error)."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None or raw == "":
        return {}, None
    if not isinstance(raw, str):
        return None, f"tool arguments must be a JSON object, got {type(raw).__name__}"
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, (
            f"tool arguments were not valid JSON ({exc}). "
            f"Send arguments as a JSON object, e.g. {{\"path\": \"foo.txt\"}}."
        )
    if not isinstance(parsed, dict):
        return None, (
            f"tool arguments must be a JSON object, got {type(parsed).__name__}. "
            f'Example: {{"path": "foo.txt"}}'
        )
    return parsed, None


def _require_str(args: dict[str, Any], key: str) -> tuple[str | None, str | None]:
    if key not in args:
        return None, f"missing required argument {key!r}"
    value = args[key]
    if value is None:
        return None, f"argument {key!r} was null, expected a string"
    if not isinstance(value, str):
        # Models sometimes send a number or a list where a string belongs.
        # Numbers are safe to stringify; structures are not what was meant.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value), None
        return None, f"argument {key!r} must be a string, got {type(value).__name__}"
    return value, None


def _coerce_timeout(value: Any) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT
    # Clamp: a model asking for a 1-hour timeout will stall the whole run.
    return max(1, min(timeout, 600))


def format_shell_result(result: Any) -> str:
    parts = [f"exit_code: {result.exit_code}"]
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        parts.append(f"--- stdout ---\n{stdout}")
    if stderr:
        parts.append(f"--- stderr ---\n{stderr}")
    if not stdout and not stderr:
        parts.append("(no output)")
    return "\n".join(parts)


def execute_tool(name: Any, arguments: Any, executor: Executor) -> str:
    """Dispatch one tool call. Returns the tool result as a string, always.

    Never raises -- malformed names, malformed JSON, bad types and sandbox
    failures all come back as error strings the model can act on.
    """
    try:
        if not isinstance(name, str) or not name:
            return f"Error: invalid tool name. Available tools: {', '.join(TOOL_NAMES)}"

        if name not in TOOL_NAMES:
            return (
                f"Error: unknown tool {name!r}. "
                f"Available tools: {', '.join(TOOL_NAMES)}"
            )

        args, err = parse_arguments(arguments)
        if err is not None:
            return f"Error: {err}"
        assert args is not None

        if name == "read_file":
            path, err = _require_str(args, "path")
            if err:
                return f"Error: {err}"
            return truncate(executor.read_file(path))

        if name == "write_file":
            path, err = _require_str(args, "path")
            if err:
                return f"Error: {err}"
            content, err = _require_str(args, "content")
            if err:
                return f"Error: {err}"
            executor.write_file(path, content)
            line_count = content.count("\n") + (1 if content else 0)
            return f"Wrote {len(content)} bytes ({line_count} lines) to {path}"

        if name == "run_shell":
            command, err = _require_str(args, "command")
            if err:
                return f"Error: {err}"
            timeout = _coerce_timeout(args.get("timeout", DEFAULT_TIMEOUT))
            result = executor.run_shell(command, timeout=timeout)
            return truncate(format_shell_result(result))

        if name == "task_complete":
            # The loop intercepts this before dispatch; handled here too so the
            # dispatcher is correct on its own.
            summary, err = _require_str(args, "summary")
            if err:
                return f"Error: {err}"
            return summary

        return f"Error: tool {name!r} is declared but not implemented"

    except SandboxError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001 - the whole point is to not escape
        return f"Error: {type(exc).__name__}: {exc}"
