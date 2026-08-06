"""Tool schemas and dispatch.

The contract for `execute_tool` is that it NEVER raises. Llama and Qwen produce
malformed tool-call JSON, invented tool names, and wrong-typed arguments often
enough that treating those as exceptions would kill the loop several times per
benchmark run. Every failure comes back as a plain error string in the tool
result so the model can see what it did wrong and correct on the next turn.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import PurePosixPath
from typing import Any

from .sandbox import DEFAULT_TIMEOUT, MAX_MATCHES, Executor, SandboxError

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
            "name": "edit_file",
            "description": (
                "Replace an exact snippet in a file. Prefer this over write_file "
                "for changes to existing files: it costs a fraction of the tokens "
                "and cannot accidentally drop the parts you did not mention. "
                "`old` must appear exactly once, including indentation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "old": {
                        "type": "string",
                        "description": "Exact text to replace, including indentation.",
                    },
                    "new": {"type": "string", "description": "Text to put in its place."},
                    "replace_all": {
                        "type": ["boolean", "string"],
                        "description": "Replace every occurrence instead of requiring exactly one.",
                    },
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": (
                "List files matching a glob, e.g. '**/*.py' or 'src/*.ts'. "
                "Faster and more reliable than shelling out to find."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern."},
                    "path": {"type": "string", "description": "Directory to search from."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search file contents for a regular expression and return matching "
                "lines with their file and line number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression."},
                    "path": {"type": "string", "description": "Directory to search from."},
                    "glob": {
                        "type": "string",
                        "description": "Only search files matching this glob, e.g. '*.py'.",
                    },
                },
                "required": ["pattern"],
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

# Providers whose schema layer cannot express a union type. Gemini's OpenAI
# endpoint converts the schema into its own OpenAPI-derived form, where `type`
# is a single string -- a list there is rejected outright, taking the whole
# request with it. Groq is the opposite case (see the timeout comment above),
# which is why the union stays canonical and gets narrowed per provider rather
# than being dropped.
SINGLE_TYPE_PROVIDERS = frozenset({"gemini"})


def tools_for(provider: str, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The tool schemas as this provider will accept them.

    Narrowing loses nothing at runtime: every union exists because models send
    the wrong scalar type, and the dispatcher coerces it either way.
    """
    tools = TOOLS if tools is None else tools
    if provider not in SINGLE_TYPE_PROVIDERS:
        return list(tools)

    narrowed = []
    for tool in tools:
        spec = copy.deepcopy(tool)
        properties = (
            spec.get("function", {}).get("parameters", {}).get("properties", {})
        )
        for schema in properties.values():
            if isinstance(schema.get("type"), list) and schema["type"]:
                # First entry is the real type; the alternates are only there to
                # stop a server-side validator rejecting a coercible value.
                schema["type"] = schema["type"][0]
        narrowed.append(spec)
    return narrowed


# --- recovering tool calls the model wrote as prose -------------------------
#
# Llama and Qwen intermittently emit a tool call as *text* in the content field
# instead of through the API's tool_calls field -- e.g.
#     <function/run_shell {"command": "ls"} </function>
# Groq's server-side parser catches most of these, but not all. When one slips
# through, the loop sees a turn with no tool calls and stops dead on step 1, so
# recovering them is the difference between a task running and a task scoring 0.

# Matches <function=name>, <function/name, <function name>, <function: name>,
# and <function(name)= -- the last observed live from llama-3.3-70b, which cost
# a whole run: no tool calls, no recovery, stopped at step 1.
_FUNCTION_TAG_RE = re.compile(
    r"<function\s*[=/:(]?\s*([A-Za-z_][\w.-]*)\s*\)?\s*[=:]?\s*>?", re.IGNORECASE
)

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


def _coerce_bool(value: Any) -> bool:
    """Models send true, "true", "True" and 1 interchangeably."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _edit_file(
    executor: Executor, path: str, old: str, new: str, replace_all: Any
) -> str:
    """Replace an exact snippet, or explain precisely why it could not.

    A silent near-miss is the dangerous failure here: if `old` does not match,
    the only safe move is to refuse and say so, because guessing at what the
    model meant would corrupt the file. The error text is written for the model
    to act on, since it is what comes back as the tool result.
    """
    if not old:
        return (
            "Error: 'old' was empty. To create a file or replace it entirely, "
            "use write_file."
        )

    content = executor.read_file(path)
    count = content.count(old)

    if count == 0:
        # Whitespace is the usual culprit, so say so rather than just "no match".
        hint = ""
        if old.strip() and old.strip() in content:
            hint = (
                " The text is present but the surrounding whitespace differs — "
                "read the file again and copy the indentation exactly."
            )
        return f"Error: 'old' text was not found in {path}.{hint}"

    if count > 1 and not _coerce_bool(replace_all):
        return (
            f"Error: 'old' text appears {count} times in {path}. Include more "
            f"surrounding context to make it unique, or pass replace_all: true."
        )

    updated = content.replace(old, new)
    executor.write_file(path, updated)

    delta = updated.count("\n") - content.count("\n")
    sign = "+" if delta > 0 else ""
    where = f"{count} occurrences" if count > 1 else "1 occurrence"
    return f"Edited {path} ({where}, {sign}{delta} lines)"


def _find_files(executor: Executor, pattern: str, root: str) -> str:
    """Glob matching applied in Python: shell globstar support varies, and
    PurePosixPath.match handles '**' the same way everywhere."""
    paths = executor.list_files(root)
    matches = [p for p in paths if PurePosixPath(p).match(pattern)]
    if not matches:
        return f"No files matching {pattern!r} under {root}"

    shown = matches[:MAX_MATCHES]
    out = "\n".join(shown)
    if len(matches) > len(shown):
        out += f"\n… {len(matches) - len(shown)} more matches"
    return out


def _search(executor: Executor, pattern: str, root: str, glob: Any) -> str:
    # Compile here so a bad pattern is a clear message rather than an empty
    # result from a grep that silently failed.
    try:
        re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regular expression: {exc}"

    hits = executor.search(
        pattern, root, glob if isinstance(glob, str) and glob else None
    )
    if not hits:
        return f"No matches for {pattern!r} under {root}"
    out = "\n".join(hits[:MAX_MATCHES])
    if len(hits) > MAX_MATCHES:
        out += f"\n… {len(hits) - MAX_MATCHES} more matches"
    return out


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

        if name == "edit_file":
            path, err = _require_str(args, "path")
            if err:
                return f"Error: {err}"
            old, err = _require_str(args, "old")
            if err:
                return f"Error: {err}"
            new, err = _require_str(args, "new")
            if err:
                return f"Error: {err}"
            return _edit_file(executor, path, old, new, args.get("replace_all"))

        if name == "find_files":
            pattern, err = _require_str(args, "pattern")
            if err:
                return f"Error: {err}"
            root = args.get("path") or "."
            return _find_files(executor, pattern, str(root))

        if name == "search":
            pattern, err = _require_str(args, "pattern")
            if err:
                return f"Error: {err}"
            root = args.get("path") or "."
            return _search(executor, pattern, str(root), args.get("glob"))

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
