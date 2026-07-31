"""A stand-in for the Groq client.

Lets the loop be tested without an API key, and -- more importantly -- lets us
script the malformed tool calls that real open models emit, which is exactly the
behaviour the dispatcher is supposed to survive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 20
    total_tokens: int = 120


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)


def tool_call(name: str, arguments: Any, call_id: str | None = None) -> FakeToolCall:
    """`arguments` may be a dict (serialized to JSON) or a raw string, so tests
    can inject deliberately broken JSON."""
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return FakeToolCall(id=call_id or f"call_{name}", function=FakeFunction(name, raw))


def turn(*calls: FakeToolCall, content: str | None = None) -> FakeResponse:
    return FakeResponse(
        choices=[FakeChoice(FakeMessage(content=content, tool_calls=list(calls) or None))]
    )


class FakeClient:
    """Replays a scripted list of responses. Records the messages it was sent so
    tests can assert on transcript shape."""

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.chat = self  # mimics client.chat.completions.create
        self.completions = self

    def create(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class ExplodingClient:
    """Always raises -- for testing the error path."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.chat = self
        self.completions = self
        self.attempts = 0

    def create(self, **kwargs: Any) -> Any:
        self.attempts += 1
        raise self.exc
