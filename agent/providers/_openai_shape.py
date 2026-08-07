"""Shared machinery for the two SDKs that speak the OpenAI wire protocol.

`openai` and `groq` are separate packages with separate exception hierarchies,
so the transports stay separate classes. What they genuinely have in common is
the *response* shape, and that reassembly is delicate enough -- streamed tool
calls arrive as fragments -- that keeping one copy is worth more than the
symmetry of four independent files.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .base import Completion, Usage, normalize_tool_call


def passthrough_fields(call: Any) -> dict[str, Any]:
    """Provider-specific fields on a tool call that must be handed back.

    Gemini 3 rejects turn two unless its `thought_signature` comes back
    unchanged. Anything the SDK did not model is preserved rather than guessed
    at, because the next such field will not be one we know about either.
    """
    if isinstance(call, dict):
        return {k: v for k, v in call.items() if k not in ("id", "type", "function")}
    return dict(getattr(call, "model_extra", None) or {})


def from_response(response: Any) -> Completion:
    """A non-streamed turn."""
    choice = response.choices[0]
    message = choice.message
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        calls.append(
            normalize_tool_call(
                call.function.name,
                call.function.arguments,
                call_id=getattr(call, "id", None),
                **passthrough_fields(call),
            )
        )
    return Completion(
        content=getattr(message, "content", None) or "",
        tool_calls=calls,
        usage=getattr(response, "usage", None),
    )


def from_stream(raw: Any, on_text: Callable[[str], None] | None = None) -> Completion:
    """A streamed turn, reassembled.

    Every branch here is a real provider behaviour, each covered by a test in
    tests/test_streaming.py:

    - fragments are keyed by `index`, because several tool calls stream
      interleaved;
    - the function name is accumulated, but an immediate repeat is skipped:
      some providers send the whole name in every delta, others chunk it;
    - usage arrives in a final chunk with **no choices**, so returning early on
      empty choices blanks the token column.
    """
    parts: list[str] = []
    fragments: dict[int, dict[str, str]] = {}
    extras: dict[int, dict[str, Any]] = {}
    usage = None

    for chunk in raw:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage
        if not getattr(chunk, "choices", None):
            continue  # usage-only chunk

        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            parts.append(text)
            if on_text:
                on_text(text)

        for fragment in getattr(delta, "tool_calls", None) or []:
            index = getattr(fragment, "index", 0) or 0
            slot = fragments.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if getattr(fragment, "id", None):
                slot["id"] = fragment.id
            function = getattr(fragment, "function", None)
            if function is not None:
                name = getattr(function, "name", None)
                if name and not slot["name"].endswith(name):
                    slot["name"] += name
                arguments = getattr(function, "arguments", None)
                if arguments:
                    slot["arguments"] += arguments
            extras.setdefault(index, {}).update(passthrough_fields(fragment))

    calls = [
        normalize_tool_call(
            slot["name"],
            slot["arguments"] or "{}",
            call_id=slot["id"] or None,
            **extras.get(index, {}),
        )
        for index, slot in sorted(fragments.items())
        if slot["name"]
    ]
    return Completion(content="".join(parts), tool_calls=calls, usage=usage)


def failed_generation_text(exc: Exception) -> str:
    """Salvage the model's reply from a schema-validation 400.

    Groq validates the model's tool arguments against our schema server-side
    and rejects the whole generation, but returns what it rejected in
    `failed_generation`. Recovering that text lets the caller run it through
    the same recovery path used for calls written as prose, instead of losing
    the task.

    Only `tool_use_failed` is salvaged. Other 400s (a bad key, an unknown
    model) must still surface as errors.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return ""
    # openai>=2 unwraps the {"error": {...}} envelope, older versions do not.
    error = body.get("error") if isinstance(body.get("error"), dict) else body
    if error.get("code") != "tool_use_failed":
        return ""
    failed = error.get("failed_generation")
    return failed if isinstance(failed, str) else ""


def usage_from(raw: Any) -> Usage | None:
    if raw is None:
        return None
    return Usage(
        prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
    )


def build_kwargs(
    model: str,
    messages: Any,
    tools: Any,
    stream: bool,
    include_usage: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "tools": list(tools),
        "tool_choice": "auto",
    }
    if not kwargs["tools"]:
        # An empty tools list is rejected by some servers; omitting it is the
        # portable way to say "no tools".
        kwargs.pop("tools")
        kwargs.pop("tool_choice")
    if stream:
        kwargs["stream"] = True
        if include_usage:
            # Without this a streamed turn reports no token usage at all.
            kwargs["stream_options"] = {"include_usage": True}
    return kwargs


def clean_messages(messages: Any) -> list[dict[str, Any]]:
    """Drop keys the OpenAI protocol does not carry.

    The transcript is the canonical record and may hold provider-specific
    fields written by another transport; sending Anthropic's block structure to
    Groq would 400.
    """
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    cleaned = []
    for message in messages:
        item = {k: v for k, v in message.items() if k in allowed}
        content = item.get("content")
        if content is not None and not isinstance(content, str):
            item["content"] = json.dumps(content, default=str)
        cleaned.append(item)
    return cleaned
