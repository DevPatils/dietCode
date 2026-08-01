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


# -- streaming ---------------------------------------------------------------


@dataclass
class FakeDeltaFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class FakeDeltaToolCall:
    index: int
    id: str | None = None
    function: FakeDeltaFunction = field(default_factory=FakeDeltaFunction)


@dataclass
class FakeDelta:
    content: str | None = None
    tool_calls: list[FakeDeltaToolCall] | None = None


@dataclass
class FakeStreamChoice:
    delta: FakeDelta


@dataclass
class FakeChunk:
    choices: list[FakeStreamChoice] = field(default_factory=list)
    usage: FakeUsage | None = None


def to_chunks(response: FakeResponse, text_chunk: int = 7) -> list[FakeChunk]:
    """Split a scripted response into deltas the way a real stream arrives.

    Content is broken into small pieces and tool arguments are split across two
    chunks with the id and name only in the first -- which is exactly the shape
    that breaks a naive reassembler.
    """
    message = response.choices[0].message
    chunks: list[FakeChunk] = []

    for i in range(0, len(message.content or ""), text_chunk):
        piece = (message.content or "")[i : i + text_chunk]
        chunks.append(FakeChunk([FakeStreamChoice(FakeDelta(content=piece))]))

    for index, call in enumerate(message.tool_calls or []):
        arguments = call.function.arguments
        split = max(1, len(arguments) // 2)
        chunks.append(
            FakeChunk(
                [
                    FakeStreamChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=index,
                                    id=call.id,
                                    function=FakeDeltaFunction(
                                        name=call.function.name,
                                        arguments=arguments[:split],
                                    ),
                                )
                            ]
                        )
                    )
                ]
            )
        )
        chunks.append(
            FakeChunk(
                [
                    FakeStreamChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=index,
                                    function=FakeDeltaFunction(
                                        arguments=arguments[split:]
                                    ),
                                )
                            ]
                        )
                    )
                ]
            )
        )

    # Usage arrives last, in a chunk with no choices at all.
    chunks.append(FakeChunk(choices=[], usage=response.usage))
    return chunks


class FakeClient:
    """Replays a scripted list of responses. Records the messages it was sent so
    tests can assert on transcript shape.

    The same script works streamed or not: when the caller asks for a stream,
    the scripted response is chopped into deltas.
    """

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.chat = self  # mimics client.chat.completions.create
        self.completions = self

    def create(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append([dict(m) for m in messages])
        self.kwargs.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if kwargs.get("stream"):
            return iter(to_chunks(nxt))
        return nxt


def sdk_error(payload: dict[str, Any]) -> Exception:
    """Build the exception the SDK builds, from a raw HTTP error body.

    Hand-constructing these hid a real bug once: openai>=2 unwraps the
    {"error": {...}} envelope and exposes the inner dict as .body, so a
    hand-made exception carrying the envelope passes while the live path fails.
    Always go through the SDK here.
    """
    import httpx
    import openai

    client = openai.OpenAI(api_key="x", base_url="https://api.groq.com/openai/v1")
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    status = 400
    response = httpx.Response(
        status,
        content=json.dumps(payload),
        headers={"content-type": "application/json"},
        request=request,
    )
    return client._make_status_error_from_response(response)


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
