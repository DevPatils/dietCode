"""The contract every transport meets.

`Completion` is the only thing the loop sees, so a provider is fully described
by how it turns OpenAI-shaped messages into one of these.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class TransportError(RuntimeError):
    """Something wrong with a provider that the user has to fix."""


@dataclass
class Completion:
    """One model turn, however it arrived.

    Streaming and non-streaming, and all four providers, normalize to this so
    the loop never has to care which one produced the turn.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Any = None


@dataclass
class Usage:
    """Token counts in the shape the loop's accumulator expects.

    Every provider names these differently; converting here means `Usage.add`
    upstream keeps working unchanged.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@runtime_checkable
class Transport(Protocol):
    """What the loop requires of a provider."""

    provider: str

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        stream: bool = False,
        on_text: Callable[[str], None] | None = None,
    ) -> Completion:
        """One model turn from OpenAI-shaped messages."""
        ...

    def list_models(self) -> list[str]:
        """Ids this provider will accept, best effort."""
        ...

    def is_transient(self, exc: Exception) -> bool:
        """Whether retrying this failure could plausibly help."""
        ...

    def is_quota_exhausted(self, exc: Exception) -> bool:
        """A daily cap, as opposed to a burst limit worth backing off for."""
        ...


# -- helpers shared by every transport ---------------------------------------

_counter = itertools.count(1)


def tool_call_id(prefix: str = "call") -> str:
    """A synthetic id, for providers that do not supply one.

    Every tool_call must be answerable by a `tool` message carrying a matching
    id, or the next request is rejected, so one is invented when absent.
    """
    return f"{prefix}_{next(_counter):06d}"


def normalize_tool_call(
    name: str, arguments: Any, call_id: str | None = None, **extra: Any
) -> dict[str, Any]:
    """Build the OpenAI-shaped tool call the loop and the transcript expect.

    `arguments` is always a JSON *string* here, matching the OpenAI wire
    format, because that is what `parse_arguments` and every existing test
    already handle.
    """
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {}, default=str)
    call: dict[str, Any] = {
        "id": call_id or tool_call_id(),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    call.update(extra)
    return call


def split_system(messages: Sequence[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Pull system messages out, for providers that take them as a parameter.

    Anthropic and Gemini both do. Several system messages are joined rather
    than dropped: the loop only sends one today, but losing an instruction
    silently is the kind of bug nobody finds.
    """
    system_parts = []
    rest = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
        else:
            rest.append(message)
    return "\n\n".join(system_parts), rest


def json_schema_for(tool: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """(name, description, parameters) from an OpenAI tool definition."""
    function = tool.get("function", tool)
    return (
        function.get("name", ""),
        function.get("description", ""),
        function.get("parameters") or {"type": "object", "properties": {}},
    )


def arguments_to_dict(arguments: Any) -> dict[str, Any]:
    """Tool-call arguments as a dict, for providers that want structured input.

    Never raises: a model that emits malformed JSON is an ordinary occurrence
    here, and the dispatcher upstream already reports it in a way the model can
    act on. Returning the raw text under a key would invent an argument, so an
    unparseable blob becomes an empty dict and the tool reports what it needed.
    """
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
